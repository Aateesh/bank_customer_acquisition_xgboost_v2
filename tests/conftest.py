"""
Shared test setup.

Every test runs with DUMMY configuration, even on a developer machine
that has a real config/.env: values set here take precedence because
python-dotenv never overrides variables that already exist. No test
connects to Snowflake or S3.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DUMMY_ENV = {
    "AWS_REGION": "eu-north-1",
    "S3_BUCKET_NAME": "ci-dummy-bucket",
    "S3_INPUT_PREFIX": "input/",
    "S3_PROCESSED_PREFIX": "processed/",
    "S3_OUTPUT_PREFIX": "output/",
    "S3_INPUT_FILE": "customer_acquisition_scoring_data.parquet",
    "S3_PROCESSED_FILE": "customer_acquisition_features.csv",
    "S3_OUTPUT_FILE": "customer_acquisition_predictions.csv",
    "SNOWFLAKE_ACCOUNT": "ci-dummy",
    "SNOWFLAKE_USER": "ci-dummy",
    "SNOWFLAKE_PASSWORD": "ci-dummy",
    "SNOWFLAKE_WAREHOUSE": "ci-dummy",
    "SNOWFLAKE_DATABASE": "ci-dummy",
    "SNOWFLAKE_SCHEMA": "ci-dummy",
    "SNOWFLAKE_ROLE": "ci-dummy",
    "SNOWFLAKE_TABLE": "ci-dummy",
    "SOURCE_DATA_FILE": "data/customer_acquisition_scoring_data.csv",
    "MODEL_FILE": "xgb_customer_acquisition_pipeline.joblib",
    "LOCAL_PROCESSED_FILE": "customer_acquisition_features.csv",
}

os.environ.update(DUMMY_ENV)
# Never let a developer's real AWS keys leak into tests.
for _name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    os.environ.pop(_name, None)

# Spark workers unpickle functions from src.* and need the repo on PYTHONPATH.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(PROJECT_ROOT), os.environ.get("PYTHONPATH")) if p
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def spark():
    """The production Spark configuration from the processing stage."""
    from src.processing.s3_to_pyspark import create_spark

    session = create_spark()
    yield session
    session.stop()


@pytest.fixture(scope="session")
def model_features():
    from src.scoring import score_model

    return score_model.get_model_feature_contract(score_model.load_model())


@pytest.fixture(scope="session")
def raw_sample(spark):
    """First 300 rows of the bundled stage-1 source data, as Spark sees them."""
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(PROJECT_ROOT / DUMMY_ENV["SOURCE_DATA_FILE"]))
        .limit(300)
    )
    return df.toDF(*[c.lower() for c in df.columns]).cache()
