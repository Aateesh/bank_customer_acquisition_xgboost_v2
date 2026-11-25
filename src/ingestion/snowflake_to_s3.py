import io

import boto3
import pandas as pd
import snowflake.connector

from src.config import (
    AWS_REGION,
    S3_BUCKET_NAME,
    S3_INPUT_PREFIX,
    S3_INPUT_FILE,
    SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_USER,
    SNOWFLAKE_PASSWORD,
    SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
    SNOWFLAKE_ROLE,
    SNOWFLAKE_TABLE,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_NAME = SNOWFLAKE_DATABASE
SCHEMA_NAME = SNOWFLAKE_SCHEMA
TABLE_NAME = SNOWFLAKE_TABLE


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SNOWFLAKE → S3 PIPELINE")
    print("=" * 60)

    print(
        f"\nSnowflake table:"
        f" {DATABASE_NAME}.{SCHEMA_NAME}.{TABLE_NAME}"
    )

    print(
        f"S3 bucket: {S3_BUCKET_NAME}"
    )

    # ========================================================
    # CONNECT TO SNOWFLAKE
    # ========================================================

    print("\nConnecting to Snowflake...")

    connection = snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE,
        role=SNOWFLAKE_ROLE,
        database=DATABASE_NAME,
        schema=SCHEMA_NAME,
    )

    print(
        "Connected to Snowflake successfully!"
    )

    try:

        # ====================================================
        # READ FROM SNOWFLAKE
        # ====================================================

        table_reference = (
            f'"{DATABASE_NAME}"'
            f'."{SCHEMA_NAME}"'
            f'."{TABLE_NAME}"'
        )

        query = f"""
            SELECT *
            FROM {table_reference}
        """

        print(
            "\nReading data from Snowflake..."
        )

        df = pd.read_sql(
            query,
            connection
        )

        print(
            f"Rows retrieved : {len(df):,}"
        )

        print(
            f"Columns        : {len(df.columns)}"
        )

        print("\nColumns:")

        print(
            list(df.columns)
        )

        if len(df) == 0:
            raise ValueError(
                "Snowflake returned zero rows."
            )

        # ====================================================
        # CONVERT TO PARQUET
        # ====================================================

        print(
            "\nConverting data to Parquet..."
        )

        parquet_buffer = io.BytesIO()

        df.to_parquet(
            parquet_buffer,
            index=False,
            engine="pyarrow"
        )

        parquet_buffer.seek(0)

        parquet_size_mb = (
            len(
                parquet_buffer.getvalue()
            )
            / (1024 * 1024)
        )

        print(
            f"Parquet size: "
            f"{parquet_size_mb:.2f} MB"
        )

        # ====================================================
        # CONNECT TO S3
        # ====================================================

        print(
            "\nConnecting to AWS S3..."
        )

        s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION
        )

        print(
            "Connected to S3 successfully!"
        )

        # ====================================================
        # BUILD S3 KEY
        # ====================================================

        s3_key = (
            f"{S3_INPUT_PREFIX.rstrip('/')}/"
            f"{S3_INPUT_FILE}"
        )

        print(
            "\nUploading to S3..."
        )

        print(
            f"Bucket : {S3_BUCKET_NAME}"
        )

        print(
            f"Key    : {s3_key}"
        )

        s3_client.upload_fileobj(
            parquet_buffer,
            S3_BUCKET_NAME,
            s3_key
        )

        print(
            "\nUpload completed successfully!"
        )

        # ====================================================
        # VERIFY S3
        # ====================================================

        print(
            "\nVerifying S3 object..."
        )

        response = s3_client.head_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )

        uploaded_size_mb = (
            response["ContentLength"]
            / (1024 * 1024)
        )

        print(
            f"Object size: "
            f"{uploaded_size_mb:.2f} MB"
        )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        print("\n" + "=" * 60)
        print(
            "SNOWFLAKE → S3 COMPLETED SUCCESSFULLY"
        )
        print("=" * 60)

        print(
            f"Snowflake rows : {len(df):,}"
        )

        print(
            f"S3 bucket      : {S3_BUCKET_NAME}"
        )

        print(
            f"S3 object      : {s3_key}"
        )

        print(
            f"File size      : "
            f"{uploaded_size_mb:.2f} MB"
        )

    finally:

        connection.close()

        print(
            "\nSnowflake connection closed."
        )


if __name__ == "__main__":
    main()