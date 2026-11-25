# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A proof of concept for productionizing a bank customer-acquisition propensity model. An XGBoost classifier predicts whether a customer responds to a marketing campaign. The model is trained in pandas/sklearn and served through PySpark. The dataset (`bank_customer_acquisition_150k.csv`) is synthetic, with deliberate missing values and outliers. `data_dictionary.csv` describes its columns.

The repo has two layers:

1. **Original training scripts (root):** `train_xgboost.py` (pandas + sklearn `ColumnTransformer` + `XGBClassifier`) and `train_pyspark_xgboost.py` (Spark-native port using `SparkXGBClassifier`, with no Python UDFs). `python_to_pyspark_mapping.md` maps pandas/sklearn operations to their PySpark equivalents.
2. **Production pipeline (`src/` + `run_pipeline.py`):** a five-stage batch pipeline that scores data with the model artifact trained by `train_xgboost.py`.

## Commands

All pipeline modules import `src.config`, so run them **as modules from the repo root**:

```bash
pip install -r requirements.txt          # Python 3.13; PySpark needs Java 17 (JAVA_HOME)

python run_pipeline.py                   # all 5 stages in sequence, each one a subprocess
python -m src.scoring.score_model        # run a single stage
python -m src.config                     # check that config/.env loads and print the values

python train_xgboost.py                  # retrain → writes xgb_customer_acquisition_pipeline.joblib
spark-submit train_pyspark_xgboost.py    # PySpark training variant

pip install -r requirements-dev.txt      # pytest + ruff
ruff check .                             # error-level rules only (pyproject.toml); never reformat pipeline code
pytest -v                                # dummy config from tests/conftest.py; never touches Snowflake/S3
pytest tests/test_model_contract.py::test_feature_contract_has_expected_size   # single test
python -m src.preflight [--cloud]        # read-only deployment self-check (used by CI and EC2 deploys)

python -m src.evaluation.create_evaluation_dataset   # builds data/customer_acquisition_test_data.csv
python -m src.data_generation.generate_scoring_data  # synthetic scoring data → data/

docker build -t marketing-ml-poc:latest .   # image tag that the Airflow DAG expects
docker compose up                            # Airflow 2.10 (LocalExecutor + Postgres) at :8080, admin/admin
```

On this Windows machine `JAVA_HOME` may point at JDK 25, and Spark tests hang under it. Use JDK 17 (`C:/Program Files/Eclipse Adoptium/jdk-17.0.17.10-hotspot`). `tests/test_spark_python_worker.py` is a legacy script that runs Spark at pytest collection time. `test.py` at the root is a scratch file.

## CI/CD (see README for full detail)

- `.github/workflows/ci-cd.yml`: test → Docker build and validate → push `sha-<commit>` + `latest` to ECR → deploy to EC2 via SSM Run Command. AWS access uses GitHub OIDC (`vars.AWS_ROLE_ARN`). Pull requests get no AWS access.
- `deploy/ec2_deploy.sh` (`deploy|rollback|run|status`) runs on EC2. **A deployment only verifies (`src.preflight --cloud`) and promotes an image. It never runs the pipeline.** Batch execution is `ec2_deploy.sh run`. State is kept in `/opt/bank-ml/{current,previous}_image`.
- On EC2, runtime config comes from SSM Parameter Store `/bank-ml/prod/<VAR>`. If you add a required env var, add it to `src/config.py`, `config/.env.example`, the DAG's `ENV_VARS`, `REQUIRED_VARS` in `ec2_deploy.sh`, and the CI dummy env (workflow + `tests/conftest.py`). Tests enforce the first three.
- Only `data/customer_acquisition_scoring_data.csv` and `data/customer_acquisition_test_data.csv` are tracked and copied into the image. Everything else in `data/` is ignored.

## Pipeline architecture

`run_pipeline.py` and `airflow/dags/ml_pipeline_dag.py` (DAG `customer_acquisition_ml_pipeline`, manual trigger only) define the same stage order. Each stage runs as `python -m <module>`. If you add or rename a stage, update both files.

1. `src.ingestion.snowflake_ingestion`: creates the Snowflake DB, schema, and table, then loads `SOURCE_DATA_FILE` (CSV) into `SNOWFLAKE_TABLE`.
2. `src.ingestion.snowflake_to_s3`: runs `SELECT *` from Snowflake and uploads the result to `s3://$S3_BUCKET_NAME/$S3_INPUT_PREFIX$S3_INPUT_FILE`.
3. `src.processing.s3_to_pyspark`: downloads the raw file, validates it, and runs feature engineering in Spark (`create_features`). Each partition is written directly to S3 with boto3 inside `foreachPartition`, in CSV parts of at most 50k rows, under `$S3_PROCESSED_PREFIX$S3_PROCESSED_FILE/`. Before writing, it deletes the existing objects under that prefix.
4. `src.scoring.score_model`: lists and downloads the processed CSV parts. It checks the feature contract against the artifact (`EXPECTED_FEATURE_COUNT = 41`), then scores in `mapPartitions`. Each partition loads the joblib artifact lazily. Predictions use threshold 0.50 and go to `$S3_OUTPUT_PREFIX/customer_acquisition_predictions/`.
5. `src.evaluation.evaluate_model`: reads `data/customer_acquisition_test_data.csv` with Spark and applies the **production** `create_features` (stage 3) and `normalize_feature_names` (stage 4). It then scores with the artifact, writes `data/model_metrics.json`, and uploads it to `$S3_OUTPUT_PREFIX/model_metrics.json`. `src/preprocessing/` was deleted on purpose. Don't restore it or reintroduce a separate pandas feature copy.

### Key cross-file contracts

- **Model artifact:** `xgb_customer_acquisition_pipeline.joblib` is a dict `{"model", "preprocessor", "numeric_features", "categorical_features", "target"}` produced by `train_xgboost.py`. Scoring and evaluation index into those keys. The features engineered in stage 3 must match `numeric_features + categorical_features`, because `train_xgboost.py` is the source of truth for feature definitions. If you change features, change them in both places and retrain.
- **Configuration:** `src/config.py` loads `config/.env` and raises if a required variable is missing, so importing any stage needs a complete `.env`. `evaluate_model.py` reads its env vars directly instead of going through `src.config`. Scoring and evaluation hardcode the model path rather than using `MODEL_FILE`. The DAG passes an explicit `ENV_VARS` allowlist into the container, so a new env var has to be added there as well.
- **Spark on Windows:** the processing and scoring modules set `PYSPARK_PYTHON`/`PYSPARK_DRIVER_PYTHON` to `sys.executable` before they import Spark. They run `local[2]` with the driver bound to `127.0.0.1`. Keep this setup, because Python workers fail on this machine without it.
- **Airflow + Docker:** the Airflow containers mount `/var/run/docker.sock`, and `DockerOperator` runs each stage in the `marketing-ml-poc:latest` image. Rebuild that image after code changes, or the DAG keeps running the old code. The Dockerfile copies `config/` into the image, but `.dockerignore` excludes `config/.env`, so runtime env comes from the DAG.

## Known issues (as of this writing)

- Training/serving skew: the production Spark `create_features` and the pandas feature logic in `train_xgboost.py` compute several features differently (`balance_to_income`, `channel_engagement`, day-of-week encoding, flag thresholds, categoricals not lower-cased). On the test set AUC is 0.747 with production features and 0.757 with training-style features. Don't "fix" this as part of unrelated work. Changing it is a modeling decision and needs retraining.

## Modeling caveats (from README)

Imputation values, outlier caps, and encoders are currently computed on the full dataset. In production they should be fitted on training data only, persisted, and reused at scoring time. A time-based split is more realistic for campaign data.
