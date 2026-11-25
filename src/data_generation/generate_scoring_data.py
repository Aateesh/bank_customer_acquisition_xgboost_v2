import os
import numpy as np
import pandas as pd

SEED = 42
ROWS = 100_000

OUTPUT_PATH = "data/customer_acquisition_scoring_data.csv"

np.random.seed(SEED)


def generate_data(rows: int) -> pd.DataFrame:

    customer_id = np.arange(200001, 200001 + rows)

    age = np.random.randint(18, 76, rows)

    annual_income = np.round(
        np.random.lognormal(
            mean=np.log(75000),
            sigma=0.45,
            size=rows
        ),
        2
    )

    annual_income = np.clip(
        annual_income,
        18000,
        300000
    )

    tenure_months = np.random.randint(1, 241, rows)

    credit_score = np.clip(
        np.round(
            np.random.normal(700, 55, rows)
        ),
        300,
        850
    ).astype(int)

    existing_products = np.random.choice(
        [1, 2, 3, 4, 5, 6],
        size=rows,
        p=[0.12, 0.25, 0.30, 0.20, 0.09, 0.04]
    )

    monthly_spend = np.round(
        annual_income / 12 *
        np.random.uniform(0.05, 0.35, rows),
        2
    )

    avg_monthly_balance = np.round(
        annual_income / 12 *
        np.random.uniform(0.5, 4.0, rows),
        2
    )

    digital_txn_ratio = np.round(
        np.random.beta(5, 2, rows),
        4
    )

    mobile_logins_30d = np.random.poisson(
        18,
        rows
    )

    branch_visits_6m = np.random.poisson(
        3,
        rows
    )

    web_visits_30d = np.random.poisson(
        12,
        rows
    )

    credit_utilization = np.round(
        np.random.beta(2.2, 5, rows),
        4
    )

    delinquency_12m = np.random.choice(
    [0, 1, 2, 3, 4],
    size=rows,
    p=[0.72, 0.16, 0.07, 0.04, 0.01])

    marketing_contacts_90d = np.random.poisson(
        4,
        rows
    )

    days_since_last_txn = np.random.randint(
        1,
        181,
        rows
    )

    customer_segment = np.random.choice(
        [
            "Mass Market",
            "Affluent",
            "High Net Worth",
            "Young Professional"
        ],
        size=rows,
        p=[0.50, 0.25, 0.10, 0.15]
    )

    acquisition_channel = np.random.choice(
        [
            "Email",
            "SMS",
            "Social Media",
            "Branch",
            "Web",
            "Call Center"
        ],
        size=rows,
        p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    )

    employment_type = np.random.choice(
        [
            "Salaried",
            "Self Employed",
            "Business Owner",
            "Retired"
        ],
        size=rows,
        p=[0.60, 0.20, 0.15, 0.05]
    )

    campaign_date = pd.to_datetime(
        np.random.choice(
            pd.date_range(
                start="2026-01-01",
                end="2026-12-31",
                freq="D"
            ),
            size=rows
        )
    )

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "annual_income": annual_income,
        "tenure_months": tenure_months,
        "credit_score": credit_score,
        "existing_products": existing_products,
        "monthly_spend": monthly_spend,
        "avg_monthly_balance": avg_monthly_balance,
        "digital_txn_ratio": digital_txn_ratio,
        "mobile_logins_30d": mobile_logins_30d,
        "branch_visits_6m": branch_visits_6m,
        "web_visits_30d": web_visits_30d,
        "credit_utilization": credit_utilization,
        "delinquency_12m": delinquency_12m,
        "marketing_contacts_90d": marketing_contacts_90d,
        "days_since_last_txn": days_since_last_txn,
        "customer_segment": customer_segment,
        "acquisition_channel": acquisition_channel,
        "employment_type": employment_type,
        "campaign_date": campaign_date
    })

    return df


def main():

    print("=" * 60)
    print("GENERATING SYNTHETIC SCORING DATA")
    print("=" * 60)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df = generate_data(ROWS)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nRows generated : {len(df):,}")
    print(f"Columns        : {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nSample:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nOutput:")
    print(os.path.abspath(OUTPUT_PATH))

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()