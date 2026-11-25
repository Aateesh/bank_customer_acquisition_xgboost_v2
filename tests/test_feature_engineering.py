"""
Critical preprocessing behaviour of the production code paths:

  stage 3  src.processing.s3_to_pyspark.create_features   (Spark)
  stage 4  src.scoring.score_model.score_partition         (Spark workers)
  stage 5  src.evaluation.evaluate_model.engineer_features (pandas)

These tests pin the CURRENT behaviour; they do not redefine features.
"""

import pandas as pd
import pytest
from pyspark.sql import functions as F

from conftest import PROJECT_ROOT
from src.processing import s3_to_pyspark as processing
from src.scoring import score_model

CAPPED = ["annual_income", "monthly_spend", "avg_monthly_balance", "web_visits_30d"]


@pytest.fixture(scope="module")
def features(raw_sample):
    df = processing.create_features(raw_sample)
    return score_model.normalize_feature_names(df).cache()


def test_bundled_source_data_matches_raw_schema(raw_sample):
    assert sorted(raw_sample.columns) == sorted(processing.EXPECTED_RAW_COLUMNS)


def test_validate_raw_data_accepts_bundled_sample(raw_sample):
    processing.validate_raw_data(raw_sample)


def test_validate_raw_data_rejects_duplicate_customer_ids(raw_sample):
    duplicated = raw_sample.limit(5).unionByName(raw_sample.limit(1))
    with pytest.raises(ValueError, match="Duplicate customer_id"):
        processing.validate_raw_data(duplicated)


def test_validate_raw_data_rejects_missing_column(raw_sample):
    with pytest.raises(ValueError, match="Missing raw columns"):
        processing.validate_raw_data(raw_sample.drop("credit_score"))


def test_spark_features_satisfy_model_contract(features, model_features):
    score_model.validate_processed_features(features, model_features)


def test_day_of_week_is_renamed_to_model_name(features):
    assert "campaign_dayofweek" in features.columns
    assert "campaign_day_of_week" not in features.columns


def test_outlier_capping_bounds_values(raw_sample, features):
    for column in CAPPED:
        lower, upper = raw_sample.approxQuantile(column, [0.01, 0.99], 0.01)
        row = features.agg(F.min(column).alias("lo"), F.max(column).alias("hi")).first()
        assert row["lo"] >= lower - 1e-9, column
        assert row["hi"] <= upper + 1e-9, column


def test_weekend_flag_matches_calendar(features):
    rows = features.select("campaign_date", "campaign_is_weekend").collect()
    for row in rows:
        if row["campaign_date"] is None:
            continue
        expected = 1 if row["campaign_date"].weekday() >= 5 else 0
        assert row["campaign_is_weekend"] == expected


def test_spark_scoring_produces_valid_propensities(features, model_features):
    scored = (
        features.select(["customer_id"] + model_features)
        .rdd.mapPartitions(score_model.score_partition)
        .collect()
    )
    assert len(scored) == features.count()
    assert len({row[0] for row in scored}) == len(scored)
    for _, propensity, prediction in scored:
        assert 0.0 <= propensity <= 1.0
        assert prediction == int(propensity >= score_model.PREDICTION_THRESHOLD)


EVAL_CSV = PROJECT_ROOT / "data" / "customer_acquisition_test_data.csv"


@pytest.fixture(scope="module")
def evaluation_features(spark):
    from src.evaluation import evaluate_model

    return evaluate_model.engineer_features(spark, EVAL_CSV)


def test_evaluation_uses_production_feature_engineering():
    from src.evaluation import evaluate_model

    assert evaluate_model.create_features is processing.create_features
    assert evaluate_model.normalize_feature_names is score_model.normalize_feature_names


def test_evaluation_feature_engineering_satisfies_model_contract(evaluation_features, model_features):
    missing = [c for c in model_features if c not in evaluation_features.columns]
    assert not missing


def test_evaluation_features_are_scoreable_by_model(evaluation_features, model_features):
    artifact = score_model.load_model()
    X = artifact["preprocessor"].transform(evaluation_features[model_features].head(1000))
    probabilities = artifact["model"].predict_proba(X)[:, 1]
    assert len(probabilities) == 1000
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_evaluation_labels_stay_aligned_with_customers(evaluation_features):
    raw = pd.read_csv(EVAL_CSV, usecols=["customer_id", "campaign_response"])
    assert len(evaluation_features) == len(raw)
    merged = raw.merge(
        evaluation_features[["customer_id", "campaign_response"]],
        on="customer_id",
        suffixes=("_raw", "_eval"),
    )
    assert len(merged) == len(raw)
    assert (merged["campaign_response_raw"] == merged["campaign_response_eval"]).all()
