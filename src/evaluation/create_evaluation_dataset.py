# src/evaluation/create_evaluation_dataset.py

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = PROJECT_ROOT / "bank_customer_acquisition_150k.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "customer_acquisition_test_data.csv"
)

TARGET = "campaign_response"
ID_COL = "customer_id"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREATING ORIGINAL MODEL TEST DATASET")
    print("=" * 60)

    print(f"Source: {SOURCE_FILE}")

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_FILE}"
        )

    # --------------------------------------------------------
    # 1. Load original dataset
    # --------------------------------------------------------

    df = pd.read_csv(SOURCE_FILE)

    print(f"Total source rows: {len(df):,}")

    # Match original training code
    df = df.drop_duplicates(
        subset=[ID_COL]
    ).copy()

    print(f"Rows after duplicate removal: {len(df):,}")

    # --------------------------------------------------------
    # 2. Separate X and y
    # --------------------------------------------------------

    y = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET])

    # --------------------------------------------------------
    # 3. EXACT ORIGINAL SPLIT
    #
    # Original training code:
    #
    # train_test_split(
    #     X,
    #     y,
    #     test_size=0.30,
    #     random_state=42,
    #     stratify=y
    # )
    #
    # Then:
    #
    # train_test_split(
    #     X_temp,
    #     y_temp,
    #     test_size=0.50,
    #     random_state=42,
    #     stratify=y_temp
    # )
    #
    # Result:
    # 70% Train
    # 15% Validation
    # 15% Test
    # --------------------------------------------------------

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

    print()
    print("REPRODUCED ORIGINAL SPLIT")
    print("-" * 40)
    print(f"Training rows    : {len(X_train):,}")
    print(f"Validation rows  : {len(X_valid):,}")
    print(f"Test rows        : {len(X_test):,}")

    # --------------------------------------------------------
    # 4. Reconstruct TEST dataset
    # --------------------------------------------------------

    test_df = X_test.copy()

    test_df[TARGET] = y_test

    # Keep original column ordering:
    # input features + target
    original_columns = df.columns.tolist()

    test_df = test_df[original_columns]

    # --------------------------------------------------------
    # 5. Validate
    # --------------------------------------------------------

    print()
    print("TEST DATASET VALIDATION")
    print("-" * 40)

    print(f"Test rows: {len(test_df):,}")
    print(f"Test columns: {len(test_df.columns)}")

    print()
    print("Target distribution:")
    print(
        test_df[TARGET]
        .value_counts()
        .sort_index()
    )

    print()
    print("Target percentage:")
    print(
        test_df[TARGET]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # Check customer IDs
    duplicate_ids = test_df[ID_COL].duplicated().sum()

    print()
    print(f"Duplicate customer IDs: {duplicate_ids}")

    if duplicate_ids != 0:
        raise ValueError(
            "Duplicate customer IDs found in test dataset."
        )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    test_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("TEST DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(test_df):,}")
    print(f"Columns: {len(test_df.columns)}")


if __name__ == "__main__":
    main()