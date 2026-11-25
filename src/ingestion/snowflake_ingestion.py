import csv

import snowflake.connector

from src.config import (
    SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_USER,
    SNOWFLAKE_PASSWORD,
    SNOWFLAKE_WAREHOUSE,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_SCHEMA,
    SNOWFLAKE_ROLE,
    SNOWFLAKE_TABLE,
    SOURCE_DATA_FILE,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_NAME = SNOWFLAKE_DATABASE
SCHEMA_NAME = SNOWFLAKE_SCHEMA
TABLE_NAME = SNOWFLAKE_TABLE
CSV_FILE = SOURCE_DATA_FILE


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

def get_connection():

    return snowflake.connector.connect(
        account=SNOWFLAKE_ACCOUNT,
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        warehouse=SNOWFLAKE_WAREHOUSE,
        role=SNOWFLAKE_ROLE,
    )


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database(cursor):

    print("\nCreating database...")

    cursor.execute(
        f"""
        CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}
        """
    )

    print(
        f"Database ready: {DATABASE_NAME}"
    )


# ============================================================
# CREATE SCHEMA
# ============================================================

def create_schema(cursor):

    print("\nCreating schema...")

    cursor.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS
        {DATABASE_NAME}.{SCHEMA_NAME}
        """
    )

    print(
        f"Schema ready: {SCHEMA_NAME}"
    )


# ============================================================
# CREATE TABLE
# ============================================================

def create_table(cursor):

    print("\nCreating table...")

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS
        {DATABASE_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
        (
            CUSTOMER_ID INTEGER,
            AGE INTEGER,
            ANNUAL_INCOME FLOAT,
            TENURE_MONTHS INTEGER,
            CREDIT_SCORE INTEGER,
            EXISTING_PRODUCTS INTEGER,
            MONTHLY_SPEND FLOAT,
            AVG_MONTHLY_BALANCE FLOAT,
            DIGITAL_TXN_RATIO FLOAT,
            MOBILE_LOGINS_30D INTEGER,
            BRANCH_VISITS_6M INTEGER,
            WEB_VISITS_30D INTEGER,
            CREDIT_UTILIZATION FLOAT,
            DELINQUENCY_12M INTEGER,
            MARKETING_CONTACTS_90D INTEGER,
            DAYS_SINCE_LAST_TXN INTEGER,
            CUSTOMER_SEGMENT VARCHAR(50),
            ACQUISITION_CHANNEL VARCHAR(50),
            EMPLOYMENT_TYPE VARCHAR(50),
            CAMPAIGN_DATE DATE
        )
        """
    )

    print(
        f"Table ready: {TABLE_NAME}"
    )


# ============================================================
# CLEAR EXISTING DATA
# ============================================================

def truncate_table(cursor):

    print("\nClearing existing table data...")

    cursor.execute(
        f"""
        TRUNCATE TABLE
        {DATABASE_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
        """
    )

    print("Existing data cleared.")


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(cursor):

    print("\nLoading CSV data...")
    print(f"Source file: {CSV_FILE}")

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.reader(file)

        # Skip header
        next(reader)

        rows = list(reader)

    print(
        f"Rows read from CSV: {len(rows):,}"
    )

    insert_query = f"""
        INSERT INTO
        {DATABASE_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
        (
            CUSTOMER_ID,
            AGE,
            ANNUAL_INCOME,
            TENURE_MONTHS,
            CREDIT_SCORE,
            EXISTING_PRODUCTS,
            MONTHLY_SPEND,
            AVG_MONTHLY_BALANCE,
            DIGITAL_TXN_RATIO,
            MOBILE_LOGINS_30D,
            BRANCH_VISITS_6M,
            WEB_VISITS_30D,
            CREDIT_UTILIZATION,
            DELINQUENCY_12M,
            MARKETING_CONTACTS_90D,
            DAYS_SINCE_LAST_TXN,
            CUSTOMER_SEGMENT,
            ACQUISITION_CHANNEL,
            EMPLOYMENT_TYPE,
            CAMPAIGN_DATE
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
    """

    cursor.executemany(
        insert_query,
        rows
    )

    print(
        f"Inserted {len(rows):,} rows"
    )


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(cursor):

    print("\nValidating Snowflake data...")

    cursor.execute(
        f"""
        SELECT COUNT(*)
        FROM {DATABASE_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
        """
    )

    count = cursor.fetchone()[0]

    print(
        f"Rows in Snowflake: {count:,}"
    )

    if count == 0:
        raise ValueError(
            "Snowflake table contains zero rows."
        )

    cursor.execute(
        f"""
        SELECT *
        FROM {DATABASE_NAME}.{SCHEMA_NAME}.{TABLE_NAME}
        LIMIT 5
        """
    )

    print("\nSample records:")

    for row in cursor.fetchall():
        print(row)

    return count


# ============================================================
# MAIN
# ============================================================

def main():

    connection = None
    cursor = None

    try:

        print("=" * 60)
        print("SNOWFLAKE INGESTION")
        print("=" * 60)

        print(f"\nDatabase : {DATABASE_NAME}")

        print(
            f"Schema   : {SCHEMA_NAME}"
        )

        print(
            f"Table    : {TABLE_NAME}"
        )

        print(
            f"CSV      : {CSV_FILE}"
        )

        print("\nConnecting to Snowflake...")

        connection = get_connection()

        cursor = connection.cursor()

        print(
            "Connected successfully!"
        )

        create_database(cursor)

        cursor.execute(
            f"USE DATABASE {DATABASE_NAME}"
        )

        create_schema(cursor)

        cursor.execute(
            f"USE SCHEMA {SCHEMA_NAME}"
        )

        create_table(cursor)

        # Important for repeatable POC runs
        truncate_table(cursor)

        load_csv(cursor)

        connection.commit()

        row_count = validate_data(cursor)

        if row_count != 100000:
            print(
                f"\nWARNING: Expected approximately "
                f"100,000 rows but found {row_count:,}"
            )

        print("\n" + "=" * 60)
        print(
            "SNOWFLAKE INGESTION COMPLETED SUCCESSFULLY"
        )
        print("=" * 60)

    except Exception as error:

        if connection:
            connection.rollback()

        print("\nERROR:")
        print(error)

        raise

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        print(
            "\nSnowflake connection closed."
        )


if __name__ == "__main__":
    main()