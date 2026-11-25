"""
PySpark version of the Bank Customer Acquisition model.

The intent is to demonstrate how common pandas/sklearn operations map
to Spark-native operations without Python UDFs.

Python/Pandas             -> PySpark
----------------------------------------------------------
read_csv                  -> spark.read.csv
drop_duplicates           -> dropDuplicates
to_datetime               -> to_date
str.strip/lower           -> trim/lower
dt.month                  -> month
dt.quarter                -> quarter
dt.dayofweek               -> dayofweek
clip                      -> when / least / greatest
log1p                     -> log1p
arithmetic                -> Spark column expressions
boolean flags             -> when
train_test_split          -> randomSplit
SimpleImputer             -> Spark ML Imputer
OneHotEncoder              -> StringIndexer + OneHotEncoder
ColumnTransformer         -> Pipeline
XGBClassifier             -> SparkXGBClassifier
predict_proba             -> probability vector
AUC                       -> BinaryClassificationEvaluator

No pandas UDFs are required in this implementation.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    trim,
    lower,
    when,
    log1p,
    month,
    quarter,
    dayofweek,
    sin,
    cos,
    lit,
    greatest,
    least,
    expr,
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from xgboost.spark import SparkXGBClassifier


DATA_PATH = r"C:\Users\Aatee\Downloads\bank_customer_acquisition_xgboost_v2\bank_customer_acquisition_150k.csv"
TARGET = "campaign_response"

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


spark = (
    SparkSession.builder
    .appName("BankCustomerAcquisitionXGBoost")
    .getOrCreate()
)


# -------------------------------------------------------
# 1. Read data
# -------------------------------------------------------
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)

df = df.dropDuplicates(["customer_id"])


# -------------------------------------------------------
# 2. Data type + categorical cleanup
# -------------------------------------------------------
df = df.withColumn(
    "campaign_date",
    col("campaign_date").cast("date")
)

for c in CATEGORICAL_FEATURES:
    df = df.withColumn(
        c,
        lower(trim(col(c)))
    )


# -------------------------------------------------------
# 3. Date features
# -------------------------------------------------------
df = (
    df
    .withColumn("campaign_month", month("campaign_date"))
    .withColumn("campaign_quarter", quarter("campaign_date"))
    .withColumn(
        "campaign_dayofweek",
        dayofweek("campaign_date")
    )
    .withColumn(
        "campaign_is_weekend",
        when(
            col("campaign_dayofweek").isin(1, 7),
            1
        ).otherwise(0)
    )
    .withColumn(
        "campaign_month_sin",
        sin(
            2 * lit(3.141592653589793)
            * col("campaign_month") / lit(12)
        )
    )
    .withColumn(
        "campaign_month_cos",
        cos(
            2 * lit(3.141592653589793)
            * col("campaign_month") / lit(12)
        )
    )
)

print("Dataframe schema after preprocessing:")
print(df.printSchema())
# -------------------------------------------------------
# 4. Outlier capping
# -------------------------------------------------------
# In production, calculate these quantiles on TRAIN only
# and persist the thresholds.
#
# The values below are examples of the Spark-native pattern.

df = (
    df
    .withColumn(
        "annual_income",
        when(col("annual_income") > 250000, 250000)
        .otherwise(col("annual_income"))
    )
    .withColumn(
        "annual_income",
        when(col("annual_income") < 18000, 18000)
        .otherwise(col("annual_income"))
    )
    .withColumn(
        "monthly_spend",
        when(col("monthly_spend") > 20000, 20000)
        .otherwise(col("monthly_spend"))
    )
    .withColumn(
        "monthly_spend",
        when(col("monthly_spend") < 100, 100)
        .otherwise(col("monthly_spend"))
    )
    .withColumn(
        "avg_monthly_balance",
        when(
            col("avg_monthly_balance") > 100000,
            100000
        ).otherwise(col("avg_monthly_balance"))
    )
)


# -------------------------------------------------------
# 5. Log transformations
# -------------------------------------------------------
df = (
    df
    .withColumn("log_income", log1p("annual_income"))
    .withColumn(
        "log_monthly_spend",
        log1p("monthly_spend")
    )
    .withColumn(
        "log_avg_balance",
        log1p("avg_monthly_balance")
    )
    .withColumn(
        "log_web_visits",
        log1p("web_visits_30d")
    )
)
print("Dataframe schema after log transformations:")

# -------------------------------------------------------
# 6. Ratio features
# -------------------------------------------------------
df = (
    df
    .withColumn(
        "spend_to_income",
        col("monthly_spend")
        / (col("annual_income") / 12 + 1)
    )
    .withColumn(
        "balance_to_income",
        col("avg_monthly_balance")
        / (col("annual_income") / 12 + 1)
    )
    .withColumn(
        "digital_engagement",
        0.6 * col("digital_txn_ratio")
        + 0.4 * (
            col("mobile_logins_30d") / 101
        )
    )
    .withColumn(
        "channel_engagement",
        col("mobile_logins_30d")
        + col("web_visits_30d")
    )
    .withColumn(
        "product_penetration",
        col("existing_products")
        / (col("tenure_months") / 12 + 1)
    )
    .withColumn(
        "delinquency_rate",
        col("delinquency_12m") / 12
    )
    .withColumn(
        "contact_pressure",
        col("marketing_contacts_90d") / 3
    )
)
print("Dataframe schema after ratio features:--------------")
# -------------------------------------------------------
# 7. Business flags
# -------------------------------------------------------
df = (
    df
    .withColumn(
        "high_value_customer",
        when(
            (col("annual_income") >= 120000)
            & (col("avg_monthly_balance") >= 10000),
            1
        ).otherwise(0)
    )
    .withColumn(
        "digital_customer",
        when(
            (col("digital_txn_ratio") >= 0.70)
            & (col("mobile_logins_30d") >= 15),
            1
        ).otherwise(0)
    )
    .withColumn(
        "credit_risk_flag",
        when(
            (col("credit_score") < 600)
            | (col("credit_utilization") > 0.80)
            | (col("delinquency_12m") >= 2),
            1
        ).otherwise(0)
    )
)


# -------------------------------------------------------
# 8. Interaction features
# -------------------------------------------------------
df = (
    df
    .withColumn(
        "income_x_credit_score",
        (col("annual_income") / 100000)
        * col("credit_score")
    )
    .withColumn(
        "digital_x_spend",
        col("digital_txn_ratio")
        * col("monthly_spend")
    )
    .withColumn(
        "tenure_x_products",
        col("tenure_months")
        * col("existing_products")
    )
)


# -------------------------------------------------------
# 9. Cast target
# -------------------------------------------------------
df = df.withColumn(
    TARGET,
    col(TARGET).cast("double")
)


# -------------------------------------------------------
# 10. Train / validation / test
# -------------------------------------------------------
train, temp = df.randomSplit(
    [0.70, 0.30],
    seed=42
)

valid, test = temp.randomSplit(
    [0.50, 0.50],
    seed=42
)


# -------------------------------------------------------
# 11. Spark-native preprocessing
# -------------------------------------------------------
DERIVED_NUMERIC = [
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

ALL_NUMERIC = NUMERIC_FEATURES + DERIVED_NUMERIC

imputed_columns = [
    f"{c}_imp" for c in ALL_NUMERIC
]

imputer = Imputer(
    inputCols=ALL_NUMERIC,
    outputCols=imputed_columns,
    strategy="median"
)

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in CATEGORICAL_FEATURES
]

encoded_columns = [
    f"{c}_ohe" for c in CATEGORICAL_FEATURES
]

encoder = OneHotEncoder(
    inputCols=[
        f"{c}_idx" for c in CATEGORICAL_FEATURES
    ],
    outputCols=encoded_columns,
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=imputed_columns + encoded_columns,
    outputCol="features",
    handleInvalid="keep"
)


# -------------------------------------------------------
# 12. Distributed XGBoost
# -------------------------------------------------------
xgb = SparkXGBClassifier(
    features_col="features",
    label_col=TARGET,
    prediction_col="prediction",
    probability_col="probability",
    rawPrediction_col="rawPrediction",

    num_workers=4,

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
    seed=42,
)


pipeline = Pipeline(
    stages=[
        imputer,
        *indexers,
        encoder,
        assembler,
        xgb,
    ]
)


# -------------------------------------------------------
# 13. Train
# -------------------------------------------------------
model = pipeline.fit(train)
print("Model training completed.")

# -------------------------------------------------------
# 14. Validation / test
# -------------------------------------------------------
valid_pred = model.transform(valid)
test_pred = model.transform(test)

evaluator = BinaryClassificationEvaluator(
    labelCol=TARGET,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

valid_auc = evaluator.evaluate(valid_pred)
test_auc = evaluator.evaluate(test_pred)

print("Validation AUC:", valid_auc)
print("Test AUC:", test_auc)


# -------------------------------------------------------
# 15. Campaign scoring
# -------------------------------------------------------
# probability is a Spark vector:
# probability[1] = probability of response.

test_pred.select(
    "customer_id",
    "probability",
    "prediction",
    TARGET
).show(20, truncate=False)


# Example: top customers by predicted response.
test_pred.select(
    "customer_id",
    "probability",
    TARGET
).orderBy(
    col("probability").desc()
).show(100)


# Save complete Spark pipeline.
model.write().overwrite().save(
    "spark_xgb_customer_acquisition_model"
)

spark.stop()
