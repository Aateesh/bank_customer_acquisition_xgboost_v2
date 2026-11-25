import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = PROJECT_ROOT / "config" / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_env(name, required=False, default=None):

    value = os.getenv(name, default)

    if required and not value:
        raise ValueError(
            f"Required environment variable '{name}' "
            f"is not configured."
        )

    return value


# ============================================================
# AWS
# ============================================================

AWS_REGION = get_env(
    "AWS_REGION",
    required=True
)

S3_BUCKET_NAME = get_env(
    "S3_BUCKET_NAME",
    required=True
)

S3_INPUT_PREFIX = get_env(
    "S3_INPUT_PREFIX",
    default="input/"
)

S3_PROCESSED_PREFIX = get_env(
    "S3_PROCESSED_PREFIX",
    default="processed/"
)

S3_OUTPUT_PREFIX = get_env(
    "S3_OUTPUT_PREFIX",
    default="output/"
)


# ============================================================
# S3 FILES
# ============================================================

S3_INPUT_FILE = get_env(
    "S3_INPUT_FILE",
    required=True
)

S3_PROCESSED_FILE = get_env(
    "S3_PROCESSED_FILE",
    required=True
)

S3_OUTPUT_FILE = get_env(
    "S3_OUTPUT_FILE",
    required=True
)


# ============================================================
# SNOWFLAKE
# ============================================================

SNOWFLAKE_ACCOUNT = get_env(
    "SNOWFLAKE_ACCOUNT",
    required=True
)

SNOWFLAKE_USER = get_env(
    "SNOWFLAKE_USER",
    required=True
)

SNOWFLAKE_PASSWORD = get_env(
    "SNOWFLAKE_PASSWORD",
    required=True
)

SNOWFLAKE_WAREHOUSE = get_env(
    "SNOWFLAKE_WAREHOUSE",
    required=True
)

SNOWFLAKE_DATABASE = get_env(
    "SNOWFLAKE_DATABASE",
    required=True
)

SNOWFLAKE_SCHEMA = get_env(
    "SNOWFLAKE_SCHEMA",
    required=True
)

SNOWFLAKE_ROLE = get_env(
    "SNOWFLAKE_ROLE",
    required=True
)

SNOWFLAKE_TABLE = get_env(
    "SNOWFLAKE_TABLE",
    required=True
)


# ============================================================
# LOCAL SOURCE DATA
# ============================================================

SOURCE_DATA_FILE = get_env(
    "SOURCE_DATA_FILE",
    required=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_FILE = get_env(
    "MODEL_FILE",
    required=True
)

MODEL_PATH = (
    PROJECT_ROOT / MODEL_FILE
)


# ============================================================
# LOCAL PROCESSED FILE
# ============================================================

LOCAL_PROCESSED_FILE = get_env(
    "LOCAL_PROCESSED_FILE",
    required=True
)


# ============================================================
# CONFIG VALIDATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)

    print("\nAWS")
    print(f"AWS_REGION          : {AWS_REGION}")
    print(f"S3_BUCKET_NAME      : {S3_BUCKET_NAME}")
    print(f"S3_INPUT_PREFIX     : {S3_INPUT_PREFIX}")
    print(f"S3_PROCESSED_PREFIX : {S3_PROCESSED_PREFIX}")
    print(f"S3_OUTPUT_PREFIX    : {S3_OUTPUT_PREFIX}")

    print("\nS3 FILES")
    print(f"S3_INPUT_FILE       : {S3_INPUT_FILE}")
    print(f"S3_PROCESSED_FILE   : {S3_PROCESSED_FILE}")
    print(f"S3_OUTPUT_FILE      : {S3_OUTPUT_FILE}")

    print("\nSNOWFLAKE")
    print(f"SNOWFLAKE_ACCOUNT   : {SNOWFLAKE_ACCOUNT}")
    print(f"SNOWFLAKE_USER      : {SNOWFLAKE_USER}")
    print(f"SNOWFLAKE_WAREHOUSE : {SNOWFLAKE_WAREHOUSE}")
    print(f"SNOWFLAKE_DATABASE  : {SNOWFLAKE_DATABASE}")
    print(f"SNOWFLAKE_SCHEMA    : {SNOWFLAKE_SCHEMA}")
    print(f"SNOWFLAKE_ROLE      : {SNOWFLAKE_ROLE}")
    print(f"SNOWFLAKE_TABLE     : {SNOWFLAKE_TABLE}")

    print("\nLOCAL")
    print(f"SOURCE_DATA_FILE    : {SOURCE_DATA_FILE}")
    print(f"LOCAL_PROCESSED_FILE: {LOCAL_PROCESSED_FILE}")

    print("\nMODEL")
    print(f"MODEL_PATH          : {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("CONFIGURATION OK")
    print("=" * 60)