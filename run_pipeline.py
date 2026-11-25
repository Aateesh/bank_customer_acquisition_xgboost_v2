import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


STAGES = [
    (
        "SNOWFLAKE INGESTION",
        "src.ingestion.snowflake_ingestion"
    ),

    (
        "SNOWFLAKE → S3",
        "src.ingestion.snowflake_to_s3"
    ),

    (
        "S3 → PYSPARK PROCESSING",
        "src.processing.s3_to_pyspark"
    ),

    (
        "MODEL SCORING",
        "src.scoring.score_model"
    ),

    (
        "MODEL EVALUATION",
        "src.evaluation.evaluate_model"
    ),
]


def run_stage(name, module):

    print("\n" + "=" * 70)
    print(f"STARTING STAGE: {name}")
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module
        ],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print("\n" + "=" * 70)
        print(f"STAGE FAILED: {name}")
        print("=" * 70)

        raise SystemExit(result.returncode)

    print("\n" + "=" * 70)
    print(f"STAGE COMPLETED: {name}")
    print("=" * 70)


def main():

    print("\n" + "=" * 70)
    print("CUSTOMER ACQUISITION ML PRODUCTIONIZATION POC")
    print("=" * 70)

    for name, module in STAGES:
        run_stage(name, module)

    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()