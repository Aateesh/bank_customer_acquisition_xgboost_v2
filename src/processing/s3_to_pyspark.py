import sys
import csv
import io,os
import uuid
from pathlib import Path

import boto3
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import TaskContext
PYTHON_EXECUTABLE = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_EXECUTABLE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXECUTABLE
# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AWS_REGION,
    S3_BUCKET_NAME,
    S3_INPUT_PREFIX,
    S3_INPUT_FILE,
    S3_PROCESSED_PREFIX,
    S3_PROCESSED_FILE,
)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_RAW_COLUMNS = [
    "customer_id",
    "age",
    "annual_income",
    "tenure_months",
    "credit_score",
    "existing_products",
    "monthly_spend",
    "avg_monthly_balance",
    "digital_txn_ratio",
    "mobile_logins_30d",
    "branch_visits_6m",
    "web_visits_30d",
    "credit_utilization",
    "delinquency_12m",
    "marketing_contacts_90d",
    "days_since_last_txn",
    "customer_segment",
    "acquisition_channel",
    "employment_type",
    "campaign_date",
]

RAW_S3_KEY = f"{S3_INPUT_PREFIX}{S3_INPUT_FILE}"

# Output will be a directory/prefix containing multiple CSV part files.
PROCESSED_S3_PREFIX = (
    f"{S3_PROCESSED_PREFIX}{S3_PROCESSED_FILE}/"
)

# Number of Spark partitions used for output.
# Increase for larger datasets.
OUTPUT_PARTITIONS = 32

# Number of rows written per S3 object from each Spark partition.
ROWS_PER_S3_FILE = 50_000


# ============================================================
# S3 CLIENT
# ============================================================

s3 = boto3.client("s3", region_name=AWS_REGION)


# ============================================================
# DELETE OLD OUTPUT
# ============================================================

def delete_s3_prefix(bucket, prefix):
    """
    Delete all existing objects under an S3 prefix.
    """

    paginator = s3.get_paginator("list_objects_v2")

    objects_to_delete = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
    ):
        for obj in page.get("Contents", []):
            objects_to_delete.append(
                {"Key": obj["Key"]}
            )

    for i in range(0, len(objects_to_delete), 1000):
        batch = objects_to_delete[i:i + 1000]

        if batch:
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": batch},
            )

    print(
        f"Deleted old S3 output: "
        f"s3://{bucket}/{prefix}"
    )


# ============================================================
# DOWNLOAD RAW PARQUET
# ============================================================

def download_raw_input():

    local_dir = PROJECT_ROOT / "data" / "s3_staging"
    local_dir.mkdir(parents=True, exist_ok=True)

    local_file = local_dir / S3_INPUT_FILE

    print("\nDownloading raw data from S3...")
    print(f"s3://{S3_BUCKET_NAME}/{RAW_S3_KEY}")

    s3.download_file(
        S3_BUCKET_NAME,
        RAW_S3_KEY,
        str(local_file),
    )

    print(f"Downloaded to: {local_file}")

    return local_file

print("\n===== PYTHON CONFIGURATION =====")
print(f"Python executable: {sys.executable}")
print(f"PYSPARK_PYTHON: {os.environ.get('PYSPARK_PYTHON')}")
print(
    f"PYSPARK_DRIVER_PYTHON: "
    f"{os.environ.get('PYSPARK_DRIVER_PYTHON')}"
)
# ============================================================
# SPARK
# ============================================================
def create_spark():
    import os
    import sys

    python_executable = sys.executable

    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

    print("\n===== CREATING SPARK SESSION =====")
    print(f"Python executable: {python_executable}")
    print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")
    print("Spark master: local[2]")
    print("Spark driver host: 127.0.0.1")
    print("Spark driver bind address: 127.0.0.1")

    spark = (
        SparkSession.builder
        .appName("CustomerAcquisitionPySparkProcessing")
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", python_executable)
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print(f"Spark version: {spark.version}")
    print("Spark session created successfully.")

    return spark

# ============================================================
# VALIDATE RAW DATA
# ============================================================

def validate_raw_data(df):

    actual_columns = df.columns

    missing_columns = [
        c for c in EXPECTED_RAW_COLUMNS
        if c not in actual_columns
    ]

    extra_columns = [
        c for c in actual_columns
        if c not in EXPECTED_RAW_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Missing raw columns: {missing_columns}"
        )

    if extra_columns:
        raise ValueError(
            f"Unexpected extra columns: {extra_columns}"
        )

    row_count = df.count()

    print(f"\nRaw row count: {row_count}")
    print(f"Raw column count: {len(actual_columns)}")

    if row_count == 0:
        raise ValueError("Raw dataset is empty.")

    null_count = (
        df.select(
            [
                F.sum(
                    F.col(c).isNull().cast("int")
                ).alias(c)
                for c in EXPECTED_RAW_COLUMNS
            ]
        )
        .collect()[0]
    )

    total_nulls = sum(
        value or 0
        for value in null_count
    )

    print(f"Total NULL values: {total_nulls}")

    duplicate_customer_ids = (
        df.groupBy("customer_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    print(
        f"Duplicate customer IDs: "
        f"{duplicate_customer_ids}"
    )

    if duplicate_customer_ids > 0:
        raise ValueError(
            "Duplicate customer_id values found."
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    print("\nStarting feature engineering...")

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    df = df.withColumn(
        "campaign_date",
        F.to_date(
            F.col("campaign_date")
        )
    )

    df = (
        df
        .withColumn(
            "campaign_month",
            F.month("campaign_date")
        )
        .withColumn(
            "campaign_quarter",
            F.quarter("campaign_date")
        )
        .withColumn(
            "campaign_day_of_week",
            F.dayofweek("campaign_date")
        )
        .withColumn(
            "campaign_is_weekend",
            F.when(
                F.dayofweek("campaign_date").isin(1, 7),
                1
            ).otherwise(0)
        )
        .withColumn(
            "campaign_month_sin",
            F.sin(
                2 * F.pi()
                * F.month("campaign_date")
                / 12
            )
        )
        .withColumn(
            "campaign_month_cos",
            F.cos(
                2 * F.pi()
                * F.month("campaign_date")
                / 12
            )
        )
    )

    # --------------------------------------------------------
    # Numeric capping
    # --------------------------------------------------------

    cap_columns = [
        "annual_income",
        "monthly_spend",
        "avg_monthly_balance",
        "web_visits_30d",
    ]

    for column in cap_columns:

        quantiles = df.approxQuantile(
            column,
            [0.01, 0.99],
            0.01,
        )

        lower = quantiles[0]
        upper = quantiles[1]

        print(
            f"{column}: "
            f"1%={lower}, "
            f"99%={upper}"
        )

        df = df.withColumn(
            column,
            F.when(
                F.col(column) < lower,
                lower
            )
            .when(
                F.col(column) > upper,
                upper
            )
            .otherwise(F.col(column))
        )

    # --------------------------------------------------------
    # Log transformations
    # --------------------------------------------------------

    df = (
        df
        .withColumn(
            "log_income",
            F.log1p(F.col("annual_income"))
        )
        .withColumn(
            "log_monthly_spend",
            F.log1p(F.col("monthly_spend"))
        )
        .withColumn(
            "log_avg_balance",
            F.log1p(F.col("avg_monthly_balance"))
        )
        .withColumn(
            "log_web_visits",
            F.log1p(F.col("web_visits_30d"))
        )
    )

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    df = (
        df
        .withColumn(
            "spend_to_income",
            F.col("monthly_spend")
            / (F.col("annual_income") / 12 + 1)
        )
        .withColumn(
            "balance_to_income",
            F.col("avg_monthly_balance")
            / (F.col("annual_income") + 1)
        )
    )

    # --------------------------------------------------------
    # Engagement
    # --------------------------------------------------------

    max_mobile_logins = (
        df
        .agg(
            F.max("mobile_logins_30d")
            .alias("max_mobile_logins")
        )
        .collect()[0]["max_mobile_logins"]
    )

    print(
        f"Maximum mobile_logins_30d: "
        f"{max_mobile_logins}"
    )

    df = (
        df
        .withColumn(
            "digital_engagement",
            (
                F.col("mobile_logins_30d")
                / F.lit(max_mobile_logins + 1)
            )
            +
            F.col("digital_txn_ratio")
        )
        .withColumn(
            "channel_engagement",
            F.when(
                F.col("acquisition_channel") == "Digital",
                1
            ).otherwise(0)
        )
        .withColumn(
            "product_penetration",
            F.col("existing_products")
            / (F.col("tenure_months") + 1)
        )
        .withColumn(
            "delinquency_rate",
            F.col("delinquency_12m")
            / (F.col("tenure_months") + 1)
        )
        .withColumn(
            "contact_pressure",
            F.col("marketing_contacts_90d")
            / (F.col("days_since_last_txn") + 1)
        )
    )

    # --------------------------------------------------------
    # Flags
    # --------------------------------------------------------

    df = (
        df
        .withColumn(
            "high_value_customer",
            F.when(
                (
                    (F.col("annual_income") >= 100000)
                    &
                    (F.col("avg_monthly_balance") >= 50000)
                ),
                1
            ).otherwise(0)
        )
        .withColumn(
            "digital_customer",
            F.when(
                (
                    (F.col("mobile_logins_30d") > 10)
                    |
                    (F.col("digital_txn_ratio") > 0.5)
                ),
                1
            ).otherwise(0)
        )
        .withColumn(
            "credit_risk_flag",
            F.when(
                (
                    (F.col("credit_utilization") > 0.7)
                    |
                    (F.col("delinquency_12m") > 0)
                ),
                1
            ).otherwise(0)
        )
    )

    # --------------------------------------------------------
    # Interactions
    # --------------------------------------------------------

    df = (
        df
        .withColumn(
            "income_x_credit_score",
            F.col("annual_income")
            * F.col("credit_score")
        )
        .withColumn(
            "digital_x_spend",
            F.col("digital_txn_ratio")
            * F.col("monthly_spend")
        )
        .withColumn(
            "tenure_x_products",
            F.col("tenure_months")
            * F.col("existing_products")
        )
    )

    print(
        f"\nFeature engineering completed."
    )

    print(
        f"Processed column count: "
        f"{len(df.columns)}"
    )

    return df


# ============================================================
# DISTRIBUTED S3 CSV UPLOAD
# ============================================================

def upload_partition_to_s3(rows):

    """
    Runs inside Spark workers.

    No Spark/Hadoop filesystem is used here.

    Each partition creates one or more CSV objects
    and uploads them directly to S3 using boto3.
    """

    import boto3
    import csv
    import io
    import uuid

    from pyspark import TaskContext

    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
    )

    partition_id = TaskContext.get().partitionId()

    batch = []
    batch_number = 0
    header = None

    def upload_batch(rows_batch, batch_no):

        if not rows_batch:
            return

        buffer = io.StringIO()

        writer = csv.writer(buffer)

        if header:
            writer.writerow(header)

        for row in rows_batch:
            writer.writerow([
                None if value is None else value
                for value in row
            ])

        object_key = (
            f"{PROCESSED_S3_PREFIX}"
            f"part-{partition_id:05d}-"
            f"{batch_no:05d}.csv"
        )

        buffer.seek(0)

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key,
            Body=buffer.getvalue(),
            ContentType="text/csv",
        )

        print(
            f"Uploaded: s3://{S3_BUCKET_NAME}/{object_key}"
        )

    for row in rows:

        if header is None:
            header = row.asDict().keys()

        batch.append(
            list(row)
        )

        if len(batch) >= ROWS_PER_S3_FILE:

            upload_batch(
                batch,
                batch_number,
            )

            batch = []
            batch_number += 1

    if batch:
        upload_batch(
            batch,
            batch_number,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER ACQUISITION PYSPARK PROCESSING")
    print("=" * 70)

    print(
        f"\nInput: "
        f"s3://{S3_BUCKET_NAME}/{RAW_S3_KEY}"
    )

    print(
        f"Output: "
        f"s3://{S3_BUCKET_NAME}/{PROCESSED_S3_PREFIX}"
    )

    # --------------------------------------------------------
    # Clean previous output
    # --------------------------------------------------------

    delete_s3_prefix(
        S3_BUCKET_NAME,
        PROCESSED_S3_PREFIX,
    )

    # --------------------------------------------------------
    # Download input
    # --------------------------------------------------------

    local_input_file = download_raw_input()

    # --------------------------------------------------------
    # Spark
    # --------------------------------------------------------

    spark = create_spark()

    try:

        print("\nReading Parquet with PySpark...")

        df = spark.read.parquet(
            str(local_input_file)
        )
        # Normalize Snowflake uppercase column names
        # to the lowercase schema used by the model.
        df = df.toDF(
            *[column.lower() for column in df.columns]
        )
        print("\n===== ACTUAL PARQUET COLUMNS AFTER NORMALIZATION =====")
        print(df.columns)
        validate_raw_data(df)

        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        df_processed = create_features(df)

        # ----------------------------------------------------
        # Cache
        # ----------------------------------------------------

        df_processed = df_processed.repartition(
            OUTPUT_PARTITIONS
        ).cache()

        processed_count = df_processed.count()

        print(
            f"\nProcessed rows: "
            f"{processed_count}"
        )

        print(
            f"Processed columns: "
            f"{len(df_processed.columns)}"
        )

        # ----------------------------------------------------
        # Distributed S3 output
        # ----------------------------------------------------

        print(
            "\nUploading processed Spark partitions "
            "directly to S3..."
        )

        df_processed.foreachPartition(
            upload_partition_to_s3
        )

        print(
            "\nSUCCESS: PySpark processing completed."
        )

        print(
            f"Processed data available at:"
        )

        print(
            f"s3://{S3_BUCKET_NAME}/"
            f"{PROCESSED_S3_PREFIX}"
        )

    finally:

        spark.stop()


if __name__ == "__main__":
    main()