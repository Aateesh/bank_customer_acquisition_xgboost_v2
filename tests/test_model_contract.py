"""The trained XGBoost artifact must match what scoring and evaluation expect."""

from src.scoring import score_model


def test_artifact_has_required_keys():
    artifact = score_model.load_model()
    for key in ("model", "preprocessor", "numeric_features", "categorical_features", "target"):
        assert key in artifact
    assert artifact["target"] == "campaign_response"


def test_feature_contract_has_expected_size(model_features):
    assert len(model_features) == score_model.EXPECTED_FEATURE_COUNT == 41


def test_artifact_feature_lists_match_preprocessor(model_features):
    artifact = score_model.load_model()
    listed = list(artifact["numeric_features"]) + list(artifact["categorical_features"])
    assert set(listed) == set(model_features)


def test_categorical_features_are_the_expected_three():
    artifact = score_model.load_model()
    assert sorted(artifact["categorical_features"]) == [
        "acquisition_channel",
        "customer_segment",
        "employment_type",
    ]
