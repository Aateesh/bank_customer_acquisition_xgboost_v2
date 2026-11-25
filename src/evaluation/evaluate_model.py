# src/evaluation/evaluate_model.py

import json
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "customer_acquisition_test_data.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "xgb_customer_acquisition_pipeline.joblib"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_metrics.json"
)


# ============================================================
# AWS CONFIG
# ============================================================

ENV_FILE = PROJECT_ROOT / "config" / ".env"


def load_env_file():

    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)


load_env_file()

import os


AWS_REGION = os.getenv(
    "AWS_REGION",
    "eu-north-1"
)

S3_BUCKET_NAME = os.getenv(
    "S3_BUCKET_NAME"
)

S3_OUTPUT_PREFIX = os.getenv(
    "S3_OUTPUT_PREFIX",
    "output/"
)

S3_METRICS_FILE = (
    f"{S3_OUTPUT_PREFIX.rstrip('/')}/model_metrics.json"
)


# ============================================================
# FEATURE ENGINEERING
#
# Evaluation uses the SAME implementation as production:
#   stage 3  src.processing.s3_to_pyspark.create_features
#   stage 4  src.scoring.score_model.normalize_feature_names
# so the metrics describe the model as it is applied in the
# production batch pipeline. (This replaces the former pandas
# copy in src/preprocessing/feature_engineering.py.)
# ============================================================

from src.processing.s3_to_pyspark import (
    create_features,
    create_spark,
)
from src.scoring.score_model import (
    normalize_feature_names,
)


TARGET = "campaign_response"


def engineer_features(spark, csv_path):
    """
    Read the raw evaluation CSV with Spark and apply the
    production feature engineering.

    The target column is carried through the (row-wise)
    transformations, so labels stay aligned with features.

    Returns a pandas DataFrame with the raw columns, the target
    and the engineered model features.
    """

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(csv_path))
    )

    # Same column-name normalization as stage 3.
    df = df.toDF(
        *[column.lower() for column in df.columns]
    )

    df = create_features(df)

    df = normalize_feature_names(df)

    return df.toPandas()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Validate files
    # --------------------------------------------------------

    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_FILE}"
        )

    # --------------------------------------------------------
    # 2. Load test data
    # --------------------------------------------------------

    print()
    print("Loading test dataset...")

    df = pd.read_csv(TEST_DATA_FILE)

    print(f"Test rows: {len(df):,}")
    print(f"Test columns: {len(df.columns)}")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    # --------------------------------------------------------
    # 3. Feature engineering (production PySpark implementation)
    # --------------------------------------------------------

    print()
    print("Applying production feature engineering (PySpark)...")

    spark = create_spark()

    try:
        features_df = engineer_features(
            spark,
            TEST_DATA_FILE,
        )
    finally:
        spark.stop()

    if len(features_df) != len(df):
        raise ValueError(
            f"Feature engineering returned {len(features_df)} rows, "
            f"expected {len(df)}."
        )

    y_true = features_df[TARGET].astype(int)

    X_features = features_df.drop(
        columns=[TARGET]
    )

    print(
        f"Feature-engineered columns: "
        f"{len(X_features.columns)}"
    )

    # --------------------------------------------------------
    # 4. Load model artifact
    # --------------------------------------------------------

    print()
    print("Loading model artifact...")

    artifact = joblib.load(MODEL_FILE)

    model = artifact["model"]
    preprocessor = artifact["preprocessor"]

    numeric_features = artifact[
        "numeric_features"
    ]

    categorical_features = artifact[
        "categorical_features"
    ]

    expected_features = (
        numeric_features
        + categorical_features
    )

    print(
        f"Expected model features: "
        f"{len(expected_features)}"
    )

    # --------------------------------------------------------
    # 5. Validate model features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X_features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            + "\n".join(missing_features)
        )

    X_model = X_features[
        expected_features
    ].copy()

    print(
        f"Scoring matrix shape: "
        f"{X_model.shape}"
    )

    # --------------------------------------------------------
    # 6. Apply trained preprocessor
    # --------------------------------------------------------

    X_transformed = (
        preprocessor.transform(X_model)
    )

    print(
        f"Transformed matrix shape: "
        f"{X_transformed.shape}"
    )

    # --------------------------------------------------------
    # 7. Generate predictions
    # --------------------------------------------------------

    print()
    print("Generating predictions...")

    probabilities = model.predict_proba(
        X_transformed
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    print("Prediction completed.")

    # --------------------------------------------------------
    # 8. Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    gini = (
        2 * roc_auc
    ) - 1

    cm = confusion_matrix(
        y_true,
        predictions
    )

    # --------------------------------------------------------
    # 9. Calculate KS
    # --------------------------------------------------------

    evaluation_df = pd.DataFrame({
        "actual": y_true.values,
        "probability": probabilities,
    })

    evaluation_df = evaluation_df.sort_values(
        "probability",
        ascending=False
    )

    total_positive = (
        evaluation_df["actual"].sum()
    )

    total_negative = (
        len(evaluation_df)
        - total_positive
    )

    if (
        total_positive > 0
        and total_negative > 0
    ):

        evaluation_df["cum_positive"] = (
            evaluation_df["actual"]
            .cumsum()
            / total_positive
        )

        evaluation_df["cum_negative"] = (
            (
                np.arange(
                    len(evaluation_df)
                )
                + 1
                - evaluation_df[
                    "actual"
                ].cumsum()
            )
            / total_negative
        )

        ks = np.max(
            np.abs(
                evaluation_df[
                    "cum_positive"
                ]
                -
                evaluation_df[
                    "cum_negative"
                ]
            )
        )

    else:
        ks = None

    # --------------------------------------------------------
    # 10. Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        predictions,
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # 11. Build metrics
    # --------------------------------------------------------

    metrics = {

        "dataset": {
            "name": "customer_acquisition_test_data",
            "rows": int(len(df)),
            "positive_cases": int(
                y_true.sum()
            ),
            "negative_cases": int(
                (y_true == 0).sum()
            ),
        },

        "classification_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },

        "ranking_metrics": {
            "roc_auc": float(roc_auc),
            "gini": float(gini),
            "ks": (
                float(ks)
                if ks is not None
                else None
            ),
        },

        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        },

        "classification_report": report,

        "threshold": 0.50,
    }

    # --------------------------------------------------------
    # 12. Print results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print()
    print("Classification Metrics")
    print("-" * 40)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print()
    print("Ranking Metrics")
    print("-" * 40)

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"Gini      : {gini:.4f}"
    )

    print(
        f"KS        : {ks:.4f}"
    )

    print()
    print("Confusion Matrix")
    print("-" * 40)

    print(
        cm
    )

    # --------------------------------------------------------
    # 13. Save JSON
    # --------------------------------------------------------

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print()
    print(
        f"Metrics saved locally:"
    )

    print(
        METRICS_FILE
    )

    # --------------------------------------------------------
    # 14. Upload to S3
    # --------------------------------------------------------

    if (
        S3_BUCKET_NAME
    ):

        print()
        print("Uploading metrics to S3...")

        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION
        )

        s3.upload_file(
            str(METRICS_FILE),
            S3_BUCKET_NAME,
            S3_METRICS_FILE
        )

        print(
            "Uploaded:"
        )

        print(
            f"s3://{S3_BUCKET_NAME}/"
            f"{S3_METRICS_FILE}"
        )

    print()
    print("=" * 60)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()