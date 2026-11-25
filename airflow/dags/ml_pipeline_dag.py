from datetime import datetime
import os

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


# Environment variables required by the ML Docker container
ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",

    "S3_BUCKET_NAME",
    "S3_INPUT_PREFIX",
    "S3_PROCESSED_PREFIX",
    "S3_OUTPUT_PREFIX",

    "S3_INPUT_FILE",
    "S3_PROCESSED_FILE",
    "S3_OUTPUT_FILE",

    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_TABLE",

    "SOURCE_DATA_FILE",
    "MODEL_FILE",
    "LOCAL_PROCESSED_FILE",
]


docker_environment = {
    key: os.environ[key]
    for key in ENV_VARS
    if os.environ.get(key)
}


COMMON_DOCKER_ARGS = {
    "image": "marketing-ml-poc:latest",
    "docker_url": "unix://var/run/docker.sock",
    "network_mode": "bridge",
    "environment": docker_environment,
    "mount_tmp_dir": False,
    "auto_remove": "success",
}


with DAG(
    dag_id="customer_acquisition_ml_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "productionization"],
) as dag:

    snowflake_ingestion = DockerOperator(
        task_id="snowflake_ingestion",
        command="python -m src.ingestion.snowflake_ingestion",
        **COMMON_DOCKER_ARGS,
    )

    snowflake_to_s3 = DockerOperator(
        task_id="snowflake_to_s3",
        command="python -m src.ingestion.snowflake_to_s3",
        **COMMON_DOCKER_ARGS,
    )

    pyspark_processing = DockerOperator(
        task_id="pyspark_processing",
        command="python -m src.processing.s3_to_pyspark",
        **COMMON_DOCKER_ARGS,
    )

    model_scoring = DockerOperator(
        task_id="model_scoring",
        command="python -m src.scoring.score_model",
        **COMMON_DOCKER_ARGS,
    )

    model_evaluation = DockerOperator(
        task_id="model_evaluation",
        command="python -m src.evaluation.evaluate_model",
        **COMMON_DOCKER_ARGS,
    )

    # Pipeline dependency order
    (
        snowflake_ingestion
        >> snowflake_to_s3
        >> pyspark_processing
        >> model_scoring
        >> model_evaluation
    )