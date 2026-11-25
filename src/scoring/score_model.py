import sys
import os
import csv
import io
from pathlib import Path

# ============================================================
# PYTHON / SPARK WORKER CONFIGURATION
# ============================================================

PYTHON_EXECUTABLE = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_EXECUTABLE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXECUTABLE


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# THIRD-PARTY IMPORTS
# ============================================================

import boto3
import joblib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ============================================================
# PROJECT CONFIG
# ============================================================

from src.config import (
    AWS_REGION,
    S3_BUCKET_NAME,
    S3_PROCESSED_PREFIX,
    S3_PROCESSED_FILE,
    S3_OUTPUT_PREFIX,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "xgb_customer_acquisition_pipeline.joblib"
)

PROCESSED_S3_PREFIX = (
    f"{S3_PROCESSED_PREFIX}"
    f"{S3_PROCESSED_FILE}/"
)

PREDICTIONS_S3_PREFIX = (
    f"{S3_OUTPUT_PREFIX}"
    "customer_acquisition_predictions/"
)

# Number of Spark partitions used for scoring.
# For the 100K POC, 32 is fine.
# This can be tuned for the production 89M-record workload.
OUTPUT_PARTITIONS = 32

# Maximum rows written per S3 prediction file.
ROWS_PER_S3_FILE = 50_000

# Classification threshold.
PREDICTION_THRESHOLD = 0.50

# Model feature count expected from the current artifact.
EXPECTED_FEATURE_COUNT = 41


# ============================================================
# MODEL ARTIFACT
# ============================================================

MODEL_ARTIFACT = None


# ============================================================
# S3 CLIENT
# ============================================================

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
)


# ============================================================
# DELETE S3 PREFIX
# ============================================================

def delete_s3_prefix(bucket, prefix):
    """
    Delete all objects under an S3 prefix.
    """

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    objects = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
    ):

        for obj in page.get(
            "Contents",
            [],
        ):

            objects.append(
                {
                    "Key": obj["Key"]
                }
            )

    for i in range(
        0,
        len(objects),
        1000,
    ):

        batch = objects[
            i:i + 1000
        ]

        if batch:

            s3.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": batch
                },
            )

    print(
        f"Deleted old output:"
        f" s3://{bucket}/{prefix}"
    )


# ============================================================
# LIST PROCESSED FILES
# ============================================================

def list_processed_files():

    files = []

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    for page in paginator.paginate(
        Bucket=S3_BUCKET_NAME,
        Prefix=PROCESSED_S3_PREFIX,
    ):

        for obj in page.get(
            "Contents",
            [],
        ):

            key = obj["Key"]

            if key.endswith(".csv"):

                files.append(
                    f"s3://{S3_BUCKET_NAME}/{key}"
                )

    if not files:

        raise FileNotFoundError(
            "No processed CSV files found in S3."
        )

    # Keep processing deterministic.
    files.sort()

    return files


# ============================================================
# DOWNLOAD PROCESSED FILES
# ============================================================

def download_processed_files(
    s3_paths,
):

    local_dir = (
        PROJECT_ROOT
        / "data"
        / "processed_staging"
    )

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_files = []

    print(
        "\nDownloading processed files..."
    )

    for index, s3_path in enumerate(
        s3_paths,
    ):

        key = s3_path.split(
            f"s3://{S3_BUCKET_NAME}/",
            1,
        )[1]

        local_file = (
            local_dir
            / f"part-{index:05d}.csv"
        )

        print(
            f"Downloading {index + 1}/"
            f"{len(s3_paths)}: {key}"
        )

        s3.download_file(
            S3_BUCKET_NAME,
            key,
            str(local_file),
        )

        local_files.append(
            str(local_file)
        )

    return local_files


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global MODEL_ARTIFACT

    if MODEL_ARTIFACT is None:

        print(
            f"\nLoading model artifact:"
            f" {MODEL_PATH}"
        )

        MODEL_ARTIFACT = joblib.load(
            MODEL_PATH
        )

        print(
            "Model artifact loaded."
        )

    return MODEL_ARTIFACT


# ============================================================
# MODEL FEATURE CONTRACT
# ============================================================

def get_model_feature_contract(
    artifact,
):
    """
    Get the exact feature contract from the trained
    preprocessing artifact.

    IMPORTANT:
    The preprocessor's feature_names_in_ is treated
    as the source of truth.

    This prevents the scoring pipeline from depending
    on manually maintained feature names.
    """

    required_keys = [
        "model",
        "preprocessor",
        "numeric_features",
        "categorical_features",
        "target",
    ]

    for key in required_keys:

        if key not in artifact:

            raise ValueError(
                f"Missing model artifact key: {key}"
            )

    preprocessor = (
        artifact["preprocessor"]
    )

    if not hasattr(
        preprocessor,
        "feature_names_in_",
    ):

        raise ValueError(
            "Model preprocessor does not contain "
            "'feature_names_in_'."
        )

    feature_columns = [
        str(column)
        for column in (
            preprocessor.feature_names_in_
        )
    ]

    numeric_features = [
        str(column)
        for column in (
            artifact["numeric_features"]
        )
    ]

    categorical_features = [
        str(column)
        for column in (
            artifact["categorical_features"]
        )
    ]

    print(
        "\n===== MODEL FEATURE CONTRACT ====="
    )

    print(
        f"Numeric features: "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_features)}"
    )

    print(
        f"Total model input features: "
        f"{len(feature_columns)}"
    )

    print(
        "\nExact features expected by "
        "preprocessor:"
    )

    for index, column in enumerate(
        feature_columns,
        start=1,
    ):

        print(
            f"{index:02d}. {column}"
        )

    if len(feature_columns) != (
        EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            "Model feature count mismatch. "
            f"Artifact expects {len(feature_columns)} "
            f"features but this POC expects "
            f"{EXPECTED_FEATURE_COUNT}."
        )

    # --------------------------------------------------------
    # Verify artifact feature lists are consistent
    # --------------------------------------------------------

    artifact_feature_columns = (
        numeric_features
        + categorical_features
    )

    if set(
        artifact_feature_columns
    ) != set(
        feature_columns
    ):

        missing_from_artifact_lists = [
            col
            for col in feature_columns
            if col not in artifact_feature_columns
        ]

        extra_in_artifact_lists = [
            col
            for col in artifact_feature_columns
            if col not in feature_columns
        ]

        raise ValueError(
            "Feature contract mismatch between "
            "preprocessor.feature_names_in_ and "
            "numeric_features + categorical_features.\n"
            f"Missing from artifact feature lists: "
            f"{missing_from_artifact_lists}\n"
            f"Extra in artifact feature lists: "
            f"{extra_in_artifact_lists}"
        )

    return feature_columns


# ============================================================
# NORMALIZE FEATURE NAMES
# ============================================================

def normalize_feature_names(df):
    """
    Normalize known naming differences between
    PySpark feature engineering and the trained model.

    Current known compatibility mapping:

        campaign_day_of_week
                    ↓
        campaign_dayofweek

    The canonical name is the name used by
    the trained model artifact.
    """

    if (
        "campaign_day_of_week" in df.columns
        and
        "campaign_dayofweek" not in df.columns
    ):

        print(
            "\nNormalizing feature name:"
        )

        print(
            "campaign_day_of_week"
            " -> "
            "campaign_dayofweek"
        )

        df = df.withColumnRenamed(
            "campaign_day_of_week",
            "campaign_dayofweek",
        )

    return df


# ============================================================
# VALIDATE PROCESSED DATA AGAINST MODEL CONTRACT
# ============================================================

def validate_processed_features(
    df,
    feature_columns,
):
    """
    Validate that Spark output contains
    exactly what the trained model requires.

    Extra non-model columns such as customer_id
    and campaign_date are allowed.
    """

    print(
        "\n===== FEATURE CONTRACT VALIDATION ====="
    )

    print(
        f"Processed DataFrame columns: "
        f"{len(df.columns)}"
    )

    print(
        f"Required model columns: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Missing model columns
    # --------------------------------------------------------

    missing_model_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_model_columns:

        raise ValueError(
            "\nMissing model columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_model_columns
            )
        )

    # --------------------------------------------------------
    # Extra columns
    # --------------------------------------------------------

    allowed_metadata_columns = {
        "customer_id",
        "campaign_date",
    }

    extra_columns = [
        column
        for column in df.columns
        if (
            column not in feature_columns
            and column not in allowed_metadata_columns
        )
    ]

    if extra_columns:

        print(
            "\nWARNING: Extra processed columns "
            "not required by model:"
        )

        for column in extra_columns:

            print(
                f"  - {column}"
            )

    # --------------------------------------------------------
    # Verify customer ID
    # --------------------------------------------------------

    if (
        "customer_id"
        not in df.columns
    ):

        raise ValueError(
            "customer_id missing "
            "from processed data."
        )

    print(
        "\nFeature contract validation: SUCCESS"
    )

    print(
        "All 41 model features are present."
    )


# ============================================================
# DISTRIBUTED MODEL SCORING
# ============================================================

def score_partition(rows):

    import joblib
    import pandas as pd
    import numpy as np

    # --------------------------------------------------------
    # Load model inside Spark worker
    # --------------------------------------------------------

    model_path = str(
        MODEL_PATH
    )

    artifact = joblib.load(
        model_path
    )

    model = artifact["model"]

    preprocessor = (
        artifact["preprocessor"]
    )

    # --------------------------------------------------------
    # CRITICAL:
    # Derive feature contract from artifact
    # inside worker as well.
    # --------------------------------------------------------

    feature_columns = [
        str(column)
        for column in (
            preprocessor.feature_names_in_
        )
    ]

    # --------------------------------------------------------
    # Convert ONLY this Spark partition
    # to Pandas.
    #
    # This is NOT df.toPandas().
    # --------------------------------------------------------

    rows_list = list(rows)

    if not rows_list:

        return

    pdf = pd.DataFrame(
        rows_list,
        columns=[
            "customer_id"
        ]
        + feature_columns,
    )

    # --------------------------------------------------------
    # Customer IDs
    # --------------------------------------------------------

    customer_ids = (
        pdf["customer_id"]
    )

    # --------------------------------------------------------
    # Model input
    #
    # IMPORTANT:
    # Select exactly the 41 features expected
    # by the trained preprocessor.
    # --------------------------------------------------------

    X = pdf[
        feature_columns
    ]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    X_processed = (
        preprocessor.transform(
            X
        )
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    propensity = (
        model.predict_proba(
            X_processed
        )[:, 1]
    )

    prediction = (
        propensity
        >= PREDICTION_THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Return prediction rows
    # --------------------------------------------------------

    for customer_id, score, pred in zip(
        customer_ids,
        propensity,
        prediction,
    ):

        yield (
            int(customer_id),
            float(score),
            int(pred),
        )


# ============================================================
# S3 PREDICTION UPLOAD
# ============================================================

def upload_prediction_partition(
    rows,
):

    import boto3
    import csv
    import io

    from pyspark import TaskContext

    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
    )

    partition_id = (
        TaskContext.get()
        .partitionId()
    )

    batch = []
    batch_number = 0

    def upload_batch(
        batch_rows,
        batch_no,
    ):

        if not batch_rows:

            return

        buffer = io.StringIO()

        writer = csv.writer(
            buffer
        )

        writer.writerow([
            "customer_id",
            "propensity_score",
            "prediction",
        ])

        for row in batch_rows:

            writer.writerow(
                row
            )

        object_key = (
            f"{PREDICTIONS_S3_PREFIX}"
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
            f"Uploaded prediction file:"
            f" s3://{S3_BUCKET_NAME}/"
            f"{object_key}"
        )

    # --------------------------------------------------------
    # Stream partition rows into batches
    # --------------------------------------------------------

    for row in rows:

        batch.append(
            row
        )

        if (
            len(batch)
            >= ROWS_PER_S3_FILE
        ):

            upload_batch(
                batch,
                batch_number,
            )

            batch = []

            batch_number += 1

    # --------------------------------------------------------
    # Upload remaining rows
    # --------------------------------------------------------

    if batch:

        upload_batch(
            batch,
            batch_number,
        )


# ============================================================
# CREATE SPARK SESSION
# ============================================================

def create_spark():
    python_executable = sys.executable

    print("\n===== PYTHON CONFIGURATION =====")
    print(f"Python executable: {python_executable}")
    print(f"PYSPARK_PYTHON: {os.environ.get('PYSPARK_PYTHON')}")
    print(f"PYSPARK_DRIVER_PYTHON: {os.environ.get('PYSPARK_DRIVER_PYTHON')}")

    spark = (
        SparkSession.builder
        .appName("CustomerAcquisitionDistributedScoring")
        .master("local[2]")
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", python_executable)
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )

    return spark


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CUSTOMER ACQUISITION "
        "DISTRIBUTED SCORING"
    )
    print("=" * 70)

    print(
        "\nProcessed input:"
    )

    print(
        f"s3://{S3_BUCKET_NAME}/"
        f"{PROCESSED_S3_PREFIX}"
    )

    print(
        "\nPrediction output:"
    )

    print(
        f"s3://{S3_BUCKET_NAME}/"
        f"{PREDICTIONS_S3_PREFIX}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    artifact = load_model()

    # --------------------------------------------------------
    # Build model feature contract
    # directly from artifact
    # --------------------------------------------------------

    feature_columns = (
        get_model_feature_contract(
            artifact
        )
    )

    # --------------------------------------------------------
    # Find processed S3 files
    # --------------------------------------------------------

    processed_files = (
        list_processed_files()
    )

    print(
        f"\nProcessed files found: "
        f"{len(processed_files)}"
    )

    # --------------------------------------------------------
    # Windows workaround
    #
    # Download distributed processed files
    # to local staging.
    #
    # Spark itself still performs scoring
    # in partitions.
    # --------------------------------------------------------

    local_files = (
        download_processed_files(
            processed_files
        )
    )

    # --------------------------------------------------------
    # Spark
    # --------------------------------------------------------

    spark = create_spark()

    try:

        # ----------------------------------------------------
        # Read processed data
        # ----------------------------------------------------

        print(
            "\nReading processed features..."
        )

        df = (
            spark.read
            .option(
                "header",
                "true",
            )
            .option(
                "inferSchema",
                "true",
            )
            .csv(local_files)
        )

        row_count = (
            df.count()
        )

        print(
            f"Processed row count: "
            f"{row_count}"
        )

        print(
            f"Processed column count: "
            f"{len(df.columns)}"
        )

        # ----------------------------------------------------
        # Normalize feature names
        # ----------------------------------------------------

        df = (
            normalize_feature_names(
                df
            )
        )

        # ----------------------------------------------------
        # Validate model feature contract
        # ----------------------------------------------------

        validate_processed_features(
            df,
            feature_columns,
        )

        # ----------------------------------------------------
        # Select ONLY:
        #
        # customer_id
        # +
        # 41 model features
        #
        # campaign_date and other extra
        # columns are intentionally excluded.
        # ----------------------------------------------------

        model_input_columns = (
            ["customer_id"]
            + feature_columns
        )

        df_model_input = (
            df.select(
                *model_input_columns
            )
        )

        print(
            "\n===== MODEL INPUT ====="
        )

        print(
            f"Model input columns: "
            f"{len(model_input_columns)}"
        )

        print(
            "customer_id + "
            f"{len(feature_columns)} "
            "model features"
        )

        # ----------------------------------------------------
        # Repartition
        # ----------------------------------------------------

        df_model_input = (
            df_model_input
            .repartition(
                OUTPUT_PARTITIONS
            )
        )

        print(
            f"\nSpark scoring partitions: "
            f"{OUTPUT_PARTITIONS}"
        )

        # ----------------------------------------------------
        # Distributed scoring
        # ----------------------------------------------------

        print(
            "\nStarting distributed "
            "model scoring..."
        )

        scored_rdd = (
            df_model_input
            .rdd
            .mapPartitions(
                score_partition
            )
        )

        # ----------------------------------------------------
        # Prediction schema
        # ----------------------------------------------------

        prediction_schema = (
            T.StructType([
                T.StructField(
                    "customer_id",
                    T.LongType(),
                    False,
                ),

                T.StructField(
                    "propensity_score",
                    T.DoubleType(),
                    False,
                ),

                T.StructField(
                    "prediction",
                    T.IntegerType(),
                    False,
                ),
            ])
        )

        scored_df = (
        spark.createDataFrame(
        scored_rdd,
        schema=prediction_schema).cache())

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        prediction_count = (
            scored_df.count()
        )

        print(
            f"\nScored rows: "
            f"{prediction_count}"
        )

        if (
            prediction_count
            != row_count
        ):

            raise ValueError(
                "Scored row count does not "
                "match processed row count."
            )

        # ----------------------------------------------------
        # Prediction distribution
        # ----------------------------------------------------

        print(
            "\nPrediction distribution:"
        )

        (
            scored_df
            .groupBy(
                "prediction"
            )
            .count()
            .orderBy(
                "prediction"
            )
            .show()
        )

        # ----------------------------------------------------
        # Propensity statistics
        # ----------------------------------------------------

        print(
            "\nPropensity statistics:"
        )

        (
            scored_df
            .select(
                F.min(
                    "propensity_score"
                ).alias(
                    "min_score"
                ),

                F.max(
                    "propensity_score"
                ).alias(
                    "max_score"
                ),

                F.avg(
                    "propensity_score"
                ).alias(
                    "mean_score"
                ),
            )
            .show()
        )

        # ----------------------------------------------------
        # Delete old prediction output
        # ----------------------------------------------------

        delete_s3_prefix(
            S3_BUCKET_NAME,
            PREDICTIONS_S3_PREFIX,
        )

        # ----------------------------------------------------
        # Upload predictions
        # directly to S3
        # ----------------------------------------------------

        print(
            "\nUploading predictions "
            "directly to S3..."
        )

        scored_df.foreachPartition(
            upload_prediction_partition
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "SUCCESS: Distributed scoring "
            "completed."
        )

        print(
            "=" * 70
        )

        print(
            "\nPredictions:"
        )

        print(
            f"s3://{S3_BUCKET_NAME}/"
            f"{PREDICTIONS_S3_PREFIX}"
        )

    finally:

        spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()