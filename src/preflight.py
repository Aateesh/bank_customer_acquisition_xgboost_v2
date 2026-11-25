"""
Deployment preflight check for the batch ML image.

    python -m src.preflight            # offline checks (CI)
    python -m src.preflight --cloud    # + read-only AWS / S3 / Snowflake checks (EC2)

This NEVER runs the batch pipeline and NEVER writes anything:
no Snowflake TRUNCATE/LOAD, no S3 delete/upload. It reuses the
production functions (create_spark, create_features, score_partition)
on a small sample of the bundled source CSV, entirely in memory.

Prints "PREFLIGHT PASSED" and exits 0 only if every check passes.
"""

import argparse
import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Spark Python workers must be able to import src.* because
# score_partition is pickled by reference from src.scoring.score_model.
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(PROJECT_ROOT), os.environ.get("PYTHONPATH")) if p
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUCCESS_MARKER = "PREFLIGHT PASSED"
FAILURE_MARKER = "PREFLIGHT FAILED"
SAMPLE_ROWS = 500


def _resolve(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ============================================================
# OFFLINE CHECKS
# ============================================================

def check_runtime():
    print(f"Python : {sys.version.split()[0]} ({sys.executable})")
    print(f"Image revision : {os.environ.get('APP_GIT_SHA', 'unknown')}")
    result = subprocess.run(
        ["java", "-version"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError("Java is not available (required by PySpark).")
    print(f"Java : {result.stderr.splitlines()[0]}")


def check_bundled_files():
    from src.config import MODEL_PATH, SOURCE_DATA_FILE
    from src.evaluation.evaluate_model import TEST_DATA_FILE

    for label, path in (
        ("model artifact", MODEL_PATH),
        ("stage 1 source data (SOURCE_DATA_FILE)", _resolve(SOURCE_DATA_FILE)),
        ("stage 5 evaluation data", TEST_DATA_FILE),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        print(f"Found {label}: {path}")


def check_stage_imports():
    import run_pipeline

    for name, module in run_pipeline.STAGES:
        importlib.import_module(module)
        print(f"Imported {module} ({name})")


def check_model_contract():
    from src.scoring import score_model

    artifact = score_model.load_model()
    return score_model.get_model_feature_contract(artifact)


def check_spark_features_and_scoring(feature_columns):
    from src.config import SOURCE_DATA_FILE
    from src.processing import s3_to_pyspark as processing
    from src.scoring import score_model

    spark = processing.create_spark()
    try:
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(_resolve(SOURCE_DATA_FILE)))
            .limit(SAMPLE_ROWS)
        )
        df = df.toDF(*[column.lower() for column in df.columns])

        processing.validate_raw_data(df)
        df = processing.create_features(df)
        df = score_model.normalize_feature_names(df)
        score_model.validate_processed_features(df, feature_columns)

        scored = (
            df.select(["customer_id"] + feature_columns)
            .rdd
            .mapPartitions(score_model.score_partition)
            .collect()
        )
    finally:
        spark.stop()

    if len(scored) != SAMPLE_ROWS:
        raise RuntimeError(
            f"Scored {len(scored)} rows, expected {SAMPLE_ROWS}."
        )
    bad = [row for row in scored if not 0.0 <= row[1] <= 1.0 or row[2] not in (0, 1)]
    if bad:
        raise RuntimeError(f"Invalid scores, e.g. {bad[:3]}")
    print(
        f"Scored {len(scored)} sample rows through Spark: "
        f"propensity min={min(r[1] for r in scored):.4f} "
        f"max={max(r[1] for r in scored):.4f}"
    )


# ============================================================
# CLOUD CHECKS (read-only)
# ============================================================

def check_aws_identity():
    import boto3

    from src.config import AWS_REGION

    if os.environ.get("AWS_ACCESS_KEY_ID"):
        raise RuntimeError(
            "Static AWS_ACCESS_KEY_ID found in the container environment. "
            "On EC2 the instance IAM role must be used instead."
        )
    try:
        arn = boto3.client("sts", region_name=AWS_REGION).get_caller_identity()["Arn"]
    except Exception as exc:
        raise RuntimeError(
            "Cannot obtain AWS credentials inside the container. On EC2 with "
            "IMDSv2, set the instance metadata hop limit to 2 or deploy with "
            f"DOCKER_NETWORK=host. Underlying error: {exc}"
        ) from exc
    # Print only the principal name, not the account id.
    print(f"AWS principal: {arn.split(':')[-1]}")


def check_s3():
    import boto3

    from src.config import AWS_REGION, S3_BUCKET_NAME, S3_INPUT_PREFIX

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.head_bucket(Bucket=S3_BUCKET_NAME)
    s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_INPUT_PREFIX, MaxKeys=1)
    print(f"S3 bucket reachable: {S3_BUCKET_NAME}")


def check_snowflake():
    from src.ingestion.snowflake_ingestion import get_connection

    connection = get_connection()  # authenticates; does not use the warehouse
    connection.close()
    print("Snowflake authentication succeeded.")


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="also run read-only AWS, S3 and Snowflake connectivity checks",
    )
    args = parser.parse_args(argv)

    failures = []

    def run(name, func, *func_args):
        print(f"\n----- {name} -----", flush=True)
        try:
            result = func(*func_args)
            print(f"[PASS] {name}", flush=True)
            return result
        except Exception as exc:  # report every failing check, not just the first
            traceback.print_exc()
            print(f"[FAIL] {name}: {exc}", flush=True)
            failures.append(name)
            return None

    run("runtime (Python + Java)", check_runtime)
    run("bundled files", check_bundled_files)
    run("pipeline stage imports", check_stage_imports)
    feature_columns = run("model artifact + feature contract", check_model_contract)
    if feature_columns:
        run(
            "Spark feature engineering + scoring (sample, in-memory)",
            check_spark_features_and_scoring,
            feature_columns,
        )
    else:
        failures.append("Spark feature engineering + scoring (skipped)")

    if args.cloud:
        run("AWS identity (IAM role)", check_aws_identity)
        run("S3 access (read-only)", check_s3)
        run("Snowflake authentication", check_snowflake)

    print("\n" + "=" * 60)
    if failures:
        print(f"{FAILURE_MARKER}: {', '.join(failures)}")
        print("=" * 60)
        return 1
    print(SUCCESS_MARKER)
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
