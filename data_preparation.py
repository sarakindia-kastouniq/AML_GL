"""
data_preparation.py
===================
Step 2 – Data Preparation
- Loads tourism.csv directly from Hugging Face dataset hub
- Performs data cleaning and feature engineering
- Splits into train / test sets (80 / 20, stratified)
- Saves splits locally and uploads them back to the HF dataset repo
"""

import os
import io
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login
from datasets import load_dataset

# ─────────────────────────────────────────────
# CONFIGURATION  ← update these before running
# ─────────────────────────────────────────────
HF_TOKEN      = os.environ.get("HF_TOKEN", "your_hf_token_here")
HF_USERNAME   = os.environ.get("HF_USERNAME", "your_hf_username_here")
DATASET_REPO  = f"{HF_USERNAME}/tourism-dataset"
TARGET_COL    = "ProdTaken"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

LOCAL_TRAIN   = "tourism_project/data/train.csv"
LOCAL_TEST    = "tourism_project/data/test.csv"


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_data_from_hf() -> pd.DataFrame:
    """Download the raw dataset from the HF dataset hub."""
    login(token=HF_TOKEN)
    print("[INFO] Loading dataset from Hugging Face …")
    ds = load_dataset(DATASET_REPO, data_files="tourism.csv", split="train")
    df = ds.to_pandas()
    print(f"[OK]  Dataset loaded – shape: {df.shape}")
    return df


def explore_data(df: pd.DataFrame):
    """Print a quick summary so observations can be noted in the notebook."""
    print("\n── Shape ──────────────────────────────")
    print(df.shape)

    print("\n── Data Types ─────────────────────────")
    print(df.dtypes)

    print("\n── Missing Values ──────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n── Target Distribution ─────────────────")
    print(df[TARGET_COL].value_counts(normalize=True).round(4))

    print("\n── Descriptive Statistics ──────────────")
    print(df.describe())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning steps:
      1. Drop the unique-identifier column (CustomerID) – not predictive.
      2. Impute numerical missing values with the column median.
      3. Impute categorical missing values with the column mode.
      4. Encode binary / ordinal categoricals.
      5. One-hot encode remaining nominal categoricals.
    """
    print("\n[INFO] Cleaning data …")

    # 1. Drop CustomerID
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
        print("[OK]  Dropped 'CustomerID'")

    # ── Identify column types ──────────────────────────────────────────
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target from the lists so we never accidentally alter it
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)
    if TARGET_COL in cat_cols:
        cat_cols.remove(TARGET_COL)

    # 2. Impute numerical columns with median
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[OK]  Imputed '{col}' with median = {median_val:.2f}")

    # 3. Impute categorical columns with mode
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"[OK]  Imputed '{col}' with mode = '{mode_val}'")

    # 4. Encode binary / ordinal categoricals manually
    binary_map = {
        "Gender": {"Male": 1, "Female": 0},
        "TypeofContact": {"Company Invited": 1, "Self Inquiry": 0},
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"[OK]  Binary-encoded '{col}'")

    # 5. One-hot encode remaining nominal categorical columns
    remaining_cat = [c for c in df.select_dtypes(include="object").columns
                     if c != TARGET_COL]
    if remaining_cat:
        df = pd.get_dummies(df, columns=remaining_cat, drop_first=True)
        print(f"[OK]  One-hot encoded: {remaining_cat}")

    print(f"[OK]  Cleaned shape: {df.shape}")
    return df


def split_and_save(df: pd.DataFrame):
    """Stratified 80/20 train-test split; saves CSVs locally."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    os.makedirs("tourism_project/data", exist_ok=True)
    train_df.to_csv(LOCAL_TRAIN, index=False)
    test_df.to_csv(LOCAL_TEST, index=False)

    print(f"\n[OK]  Train set saved – shape: {train_df.shape}")
    print(f"[OK]  Test  set saved – shape: {test_df.shape}")

    # Observation: class balance in splits
    print("\n── Train target distribution ──")
    print(train_df[TARGET_COL].value_counts(normalize=True).round(4))
    print("── Test target distribution ───")
    print(test_df[TARGET_COL].value_counts(normalize=True).round(4))

    return train_df, test_df


def upload_splits_to_hf():
    """Upload train.csv and test.csv back to the HF dataset repo."""
    api = HfApi()
    for local_path, hf_path in [(LOCAL_TRAIN, "train.csv"), (LOCAL_TEST, "test.csv")]:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=hf_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        print(f"[OK]  Uploaded {hf_path} to HF dataset: {DATASET_REPO}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 2 – DATA PREPARATION")
    print("=" * 60)

    # Load raw data from HF
    df = load_data_from_hf()

    # Quick EDA
    explore_data(df)

    # Clean
    df_clean = clean_data(df)

    # Split and save locally
    train_df, test_df = split_and_save(df_clean)

    # Upload splits back to HF
    upload_splits_to_hf()

    print("\n[DONE] Data preparation complete.")
    print(f"       Train/Test splits available at: "
          f"https://huggingface.co/datasets/{DATASET_REPO}")
