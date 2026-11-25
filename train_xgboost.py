"""
Bank Customer Acquisition Propensity Model
===========================================

This version intentionally contains common real-world preprocessing patterns
that can be translated from pandas/sklearn to PySpark:

1. Data type conversion
2. Duplicate handling
3. Missing-value profiling
4. Numeric median imputation
5. Outlier capping / winsorization
6. Log transformation
7. Ratio features
8. Interaction features
9. Date feature extraction
10. Categorical normalization
11. One-hot encoding
12. Stratified train/validation/test split
13. Class imbalance handling
14. XGBoost training
15. AUC / Gini / KS evaluation
16. Feature importance
17. Campaign propensity scoring
18. Top-decile campaign selection

Install:
    pip install pandas numpy scikit-learn xgboost joblib
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


DATA_PATH = "bank_customer_acquisition_150k.csv"

TARGET = "campaign_response"
ID_COL = "customer_id"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "tenure_months",
    "credit_score",
    "existing_products",
    "monthly_spend",
    "avg_monthly_balance",
    "digital_txn_ratio",
    "mobile_logins_30d",
    "branch_visits_6m",
    "web_visits_30d",
    "credit_utilization",
    "delinquency_12m",
    "marketing_contacts_90d",
    "days_since_last_txn",
]

CATEGORICAL_FEATURES = [
    "customer_segment",
    "acquisition_channel",
    "employment_type",
]

DATE_FEATURE = "campaign_date"


# -------------------------------------------------------
# 1. Load + basic data quality
# -------------------------------------------------------
df = pd.read_csv(DATA_PATH)

df = df.drop_duplicates(subset=[ID_COL]).copy()

df[DATE_FEATURE] = pd.to_datetime(
    df[DATE_FEATURE],
    errors="coerce"
)

# Standardize categorical strings.
for col in CATEGORICAL_FEATURES:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"": np.nan, "nan": np.nan})
    )


# -------------------------------------------------------
# 2. Date feature engineering
# -------------------------------------------------------
df["campaign_month"] = df[DATE_FEATURE].dt.month
df["campaign_quarter"] = df[DATE_FEATURE].dt.quarter
df["campaign_dayofweek"] = df[DATE_FEATURE].dt.dayofweek
df["campaign_is_weekend"] = (
    df["campaign_dayofweek"] >= 5
).astype(int)

# Cyclic representation is useful for seasonal variables.
df["campaign_month_sin"] = np.sin(
    2 * np.pi * df["campaign_month"] / 12
)
df["campaign_month_cos"] = np.cos(
    2 * np.pi * df["campaign_month"] / 12
)


# -------------------------------------------------------
# 3. Numeric outlier capping
# -------------------------------------------------------
# Fit caps from the training data in a production implementation.
# For this standalone example, use robust percentile caps.
CAP_COLS = [
    "annual_income",
    "monthly_spend",
    "avg_monthly_balance",
    "web_visits_30d",
]

for col in CAP_COLS:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=low, upper=high)


# -------------------------------------------------------
# 4. Log transformations
# -------------------------------------------------------
df["log_income"] = np.log1p(df["annual_income"])
df["log_monthly_spend"] = np.log1p(df["monthly_spend"])
df["log_avg_balance"] = np.log1p(df["avg_monthly_balance"])
df["log_web_visits"] = np.log1p(df["web_visits_30d"])


# -------------------------------------------------------
# 5. Banking/business ratio features
# -------------------------------------------------------
df["spend_to_income"] = (
    df["monthly_spend"] / (df["annual_income"] / 12 + 1)
)

df["balance_to_income"] = (
    df["avg_monthly_balance"] / (df["annual_income"] / 12 + 1)
)

df["digital_engagement"] = (
    0.6 * df["digital_txn_ratio"]
    + 0.4 * (
        df["mobile_logins_30d"]
        / (df["mobile_logins_30d"].max() + 1)
    )
)

df["channel_engagement"] = (
    df["mobile_logins_30d"]
    + df["web_visits_30d"]
)

df["product_penetration"] = (
    df["existing_products"] / (df["tenure_months"] / 12 + 1)
)

df["delinquency_rate"] = (
    df["delinquency_12m"] / 12
)

df["contact_pressure"] = (
    df["marketing_contacts_90d"] / 3
)


# -------------------------------------------------------
# 6. Business-rule flags
# -------------------------------------------------------
df["high_value_customer"] = (
    (df["annual_income"] >= 120000)
    & (df["avg_monthly_balance"] >= 10000)
).astype(int)

df["digital_customer"] = (
    (df["digital_txn_ratio"] >= 0.70)
    & (df["mobile_logins_30d"] >= 15)
).astype(int)

df["credit_risk_flag"] = (
    (df["credit_score"] < 600)
    | (df["credit_utilization"] > 0.80)
    | (df["delinquency_12m"] >= 2)
).astype(int)


# -------------------------------------------------------
# 7. Interaction features
# -------------------------------------------------------
df["income_x_credit_score"] = (
    df["annual_income"] / 100000
    * df["credit_score"]
)

df["digital_x_spend"] = (
    df["digital_txn_ratio"]
    * df["monthly_spend"]
)

df["tenure_x_products"] = (
    df["tenure_months"]
    * df["existing_products"]
)


# -------------------------------------------------------
# 8. Final model columns
# -------------------------------------------------------
DERIVED_NUMERIC_FEATURES = [
    "campaign_month",
    "campaign_quarter",
    "campaign_dayofweek",
    "campaign_is_weekend",
    "campaign_month_sin",
    "campaign_month_cos",
    "log_income",
    "log_monthly_spend",
    "log_avg_balance",
    "log_web_visits",
    "spend_to_income",
    "balance_to_income",
    "digital_engagement",
    "channel_engagement",
    "product_penetration",
    "delinquency_rate",
    "contact_pressure",
    "high_value_customer",
    "digital_customer",
    "credit_risk_flag",
    "income_x_credit_score",
    "digital_x_spend",
    "tenure_x_products",
]

ALL_NUMERIC = NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES

X = df[ALL_NUMERIC + CATEGORICAL_FEATURES].copy()
y = df[TARGET].astype(int)


# -------------------------------------------------------
# 9. Train / validation / test split
# -------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


# -------------------------------------------------------
# 10. Preprocessing pipeline
# -------------------------------------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            ALL_NUMERIC,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ]
)


# -------------------------------------------------------
# 11. XGBoost
# -------------------------------------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=5,
    reg_alpha=0.10,
    reg_lambda=1.00,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

# Fit preprocessing separately so the same fitted preprocessing
# is reused during validation and scoring.
X_train_p = preprocessor.fit_transform(X_train)
X_valid_p = preprocessor.transform(X_valid)
X_test_p = preprocessor.transform(X_test)

model.fit(
    X_train_p,
    y_train,
    eval_set=[(X_valid_p, y_valid)],
    verbose=False,
)


# -------------------------------------------------------
# 12. Model evaluation
# -------------------------------------------------------
def calculate_ks(y_true, y_score):
    data = pd.DataFrame({
        "y": np.asarray(y_true),
        "score": np.asarray(y_score),
    }).sort_values("score", ascending=False)

    total_bad = data["y"].sum()
    total_good = len(data) - total_bad

    if total_bad == 0 or total_good == 0:
        return np.nan

    data["cum_bad"] = (
        data["y"].cumsum() / total_bad
    )

    data["cum_good"] = (
        (np.arange(len(data)) + 1 - data["y"].cumsum())
        / total_good
    )

    return np.max(
        np.abs(data["cum_bad"] - data["cum_good"])
    )


def evaluate(model, X_data, y_data, label):
    score = model.predict_proba(X_data)[:, 1]
    pred = (score >= 0.50).astype(int)

    auc = roc_auc_score(y_data, score)
    gini = 2 * auc - 1
    ks = calculate_ks(y_data, score)

    print(f"\n===== {label} =====")
    print(f"AUC       : {auc:.4f}")
    print(f"Gini      : {gini:.4f}")
    print(f"KS        : {ks:.4f}")
    print(
        f"Accuracy  : "
        f"{accuracy_score(y_data, pred):.4f}"
    )
    print(
        f"Precision : "
        f"{precision_score(y_data, pred, zero_division=0):.4f}"
    )
    print(
        f"Recall    : "
        f"{recall_score(y_data, pred, zero_division=0):.4f}"
    )
    print(confusion_matrix(y_data, pred))

    return score


valid_score = evaluate(
    model, X_valid_p, y_valid, "VALIDATION"
)

test_score = evaluate(
    model, X_test_p, y_test, "TEST"
)


# -------------------------------------------------------
# 13. Feature importance
# -------------------------------------------------------
feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False
)

print("\n===== TOP FEATURES =====")
print(importance.head(25).to_string(index=False))

importance.to_csv(
    "feature_importance.csv",
    index=False
)


# -------------------------------------------------------
# 14. Campaign scoring
# -------------------------------------------------------
campaign_output = df.loc[
    X_test.index,
    [ID_COL]
].copy()

campaign_output["propensity_score"] = test_score

# Business targeting example:
# top 10% propensity customers.
cutoff = campaign_output[
    "propensity_score"
].quantile(0.90)

campaign_output["campaign_flag"] = (
    campaign_output["propensity_score"] >= cutoff
).astype(int)

campaign_output = campaign_output.sort_values(
    "propensity_score",
    ascending=False
)

campaign_output.to_csv(
    "campaign_scored_customers.csv",
    index=False
)

print("\nCampaign cutoff:", cutoff)
print(
    "Customers selected:",
    campaign_output["campaign_flag"].sum()
)


# -------------------------------------------------------
# 15. Save complete artifact
# -------------------------------------------------------
artifact = {
    "model": model,
    "preprocessor": preprocessor,
    "numeric_features": ALL_NUMERIC,
    "categorical_features": CATEGORICAL_FEATURES,
    "target": TARGET,
}

joblib.dump(
    artifact,
    "xgb_customer_acquisition_pipeline.joblib"
)

print("\nModel artifact saved.")
