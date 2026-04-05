"""
DriftGuard — Real Data Pipeline
Prepares loan_approval_dataset.csv into 6 monthly batches
Injects drift using BankChurners statistical properties

Usage:
    python scripts/prepare_real_data.py
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "datasets")   # put your CSVs here
os.makedirs(DATA_DIR, exist_ok=True)

LOAN_FILE  = os.path.join(UPLOADS_DIR, "loan_approval_dataset.csv")
CHURN_FILE = os.path.join(UPLOADS_DIR, "BankChurners.csv")


# ─────────────────────────────────────────
# STEP 1: Load & Clean Loan Dataset
# ─────────────────────────────────────────
def load_loan_data() -> pd.DataFrame:
    df = pd.read_csv(LOAN_FILE)
    df.columns = df.columns.str.strip()

    # Clean string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Binary encode target
    df["default_label"] = (df["loan_status"] == "Rejected").astype(int)

    # Rename for consistency
    df = df.rename(columns={
        "no_of_dependents":          "num_dependents",
        "income_annum":              "income_annum",
        "loan_amount":               "loan_amount",
        "loan_term":                 "loan_term",
        "cibil_score":               "cibil_score",
        "residential_assets_value":  "residential_assets",
        "commercial_assets_value":   "commercial_assets",
        "luxury_assets_value":       "luxury_assets",
        "bank_asset_value":          "bank_assets",
    })

    # Drop original ID and status cols
    df = df.drop(columns=["loan_id", "loan_status"])

    print(f"  Loan dataset loaded: {len(df)} rows")
    print(f"  Default (Rejected) rate: {df['default_label'].mean():.2%}")
    print(f"  Features: {list(df.columns)}")
    return df


# ─────────────────────────────────────────
# STEP 2: Extract BankChurners Drift Stats
# ─────────────────────────────────────────
def extract_churn_drift_stats() -> dict:
    df = pd.read_csv(CHURN_FILE)

    # These stats will be used to shift loan distributions
    # simulating customer base change (bank's portfolio shifts)
    stats = {
        "high_credit_limit_pct":    (df["Credit_Limit"] > 10000).mean(),
        "high_transaction_pct":     (df["Total_Trans_Amt"] > 5000).mean(),
        "high_utilization_pct":     (df["Avg_Utilization_Ratio"] > 0.5).mean(),
        "avg_customer_age":         df["Customer_Age"].mean(),
        "inactive_pct":             (df["Months_Inactive_12_mon"] >= 3).mean(),
        "attrition_rate":           (df["Attrition_Flag"] == "Attrited Customer").mean(),
    }

    print(f"\n  BankChurners drift stats extracted:")
    for k, v in stats.items():
        print(f"    {k}: {v:.4f}")

    return stats


# ─────────────────────────────────────────
# STEP 3: Split into 6 Monthly Batches
# ─────────────────────────────────────────
def split_into_months(df: pd.DataFrame) -> list:
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    total       = len(df_shuffled)
    per_month   = total // 6
    months      = []

    for i in range(6):
        start = i * per_month
        end   = (i + 1) * per_month if i < 5 else total
        months.append(df_shuffled.iloc[start:end].copy())

    print(f"\n  Split into 6 months:")
    for i, m in enumerate(months):
        print(f"    Month {i+1}: {len(m)} rows | default rate: {m['default_label'].mean():.2%}")

    return months


# ─────────────────────────────────────────
# STEP 4: Inject Drift into Batches
# ─────────────────────────────────────────
def inject_data_drift(df: pd.DataFrame, churn_stats: dict) -> pd.DataFrame:
    """
    DATA DRIFT — Month 3
    Input feature distributions shift.
    High-value customers enter the portfolio.
    Relationship between features and outcome stays same.
    """
    df = df.copy()
    n  = len(df)

    # Shift income upward (wealthier customers entering)
    income_shift = np.random.normal(1.4, 0.1, n)
    df["income_annum"] = (df["income_annum"] * income_shift).astype(int)

    # Shift loan amounts upward
    loan_shift = np.random.normal(1.35, 0.1, n)
    df["loan_amount"] = (df["loan_amount"] * loan_shift).astype(int)

    # CIBIL scores shift slightly higher (better credit customers)
    cibil_shift = np.random.normal(30, 10, n)
    df["cibil_score"] = (df["cibil_score"] + cibil_shift).clip(300, 900).astype(int)

    # Asset values increase
    df["residential_assets"] = (df["residential_assets"] * 1.3).astype(int)
    df["luxury_assets"]       = (df["luxury_assets"] * 1.5).astype(int)

    df["drift_type"] = "data_drift"
    return df


def inject_concept_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    CONCEPT DRIFT — Month 5 & 6
    The RELATIONSHIP between features and outcome changes.
    High CIBIL score customers now default more (economic shock).
    Low income customers now get approved more (policy change).
    """
    df   = df.copy()
    n    = len(df)

    # Flip a portion of high-CIBIL approved loans to rejected
    # Simulates: economic downturn, even good customers struggle
    high_cibil_approved = df[
        (df["cibil_score"] > 700) &
        (df["default_label"] == 0)
    ].index
    flip_count = int(len(high_cibil_approved) * 0.35)
    flip_idx   = np.random.choice(high_cibil_approved, flip_count, replace=False)
    df.loc[flip_idx, "default_label"] = 1

    # Flip a portion of low-income rejected loans to approved
    # Simulates: new government loan scheme for low-income
    low_income_rejected = df[
        (df["income_annum"] < df["income_annum"].quantile(0.3)) &
        (df["default_label"] == 1)
    ].index
    flip_count2 = int(len(low_income_rejected) * 0.4)
    flip_idx2   = np.random.choice(low_income_rejected, flip_count2, replace=False)
    df.loc[flip_idx2, "default_label"] = 0

    df["drift_type"] = "concept_drift"
    return df


# ─────────────────────────────────────────
# STEP 5: Add Dates & Save
# ─────────────────────────────────────────
def add_dates(df: pd.DataFrame, month: int) -> pd.DataFrame:
    df = df.copy()
    start = datetime(2024, month, 1)
    df["date"]  = [
        start + timedelta(days=int(np.random.randint(0, 28)))
        for _ in range(len(df))
    ]
    df["month"] = month
    return df


def save_month(df: pd.DataFrame, month: int) -> None:
    path = os.path.join(DATA_DIR, f"month_{month}.csv")
    df.to_csv(path, index=False)
    print(f"  ✅ Month {month} saved → {len(df)} rows | "
          f"default: {df['default_label'].mean():.2%} | "
          f"drift: {df.get('drift_type', pd.Series(['none'])).iloc[0]}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  DriftGuard — Real Data Pipeline")
    print("  Source: loan_approval_dataset.csv + BankChurners.csv")
    print("=" * 60)

    print("\n[1/5] Loading loan approval dataset...")
    df_loan = load_loan_data()

    print("\n[2/5] Extracting BankChurners drift statistics...")
    churn_stats = extract_churn_drift_stats()

    print("\n[3/5] Splitting into 6 monthly batches...")
    months = split_into_months(df_loan)

    print("\n[4/5] Injecting drift into specific months...")
    monthly_data = []
    for i, df_month in enumerate(months):
        month_num = i + 1

        if month_num in [1, 2]:
            df_month["drift_type"] = "none"         # Clean baseline
        elif month_num == 3:
            df_month = inject_data_drift(df_month, churn_stats)   # DATA DRIFT
        elif month_num == 4:
            df_month["drift_type"] = "none"         # Partial recovery
        elif month_num in [5, 6]:
            df_month = inject_concept_drift(df_month)             # CONCEPT DRIFT

        df_month = add_dates(df_month, month_num)
        monthly_data.append(df_month)

    print("\n[5/5] Saving monthly files...")
    for i, df_month in enumerate(monthly_data):
        save_month(df_month, i + 1)

    # Save full dataset
    full_df = pd.concat(monthly_data, ignore_index=True)
    full_path = os.path.join(DATA_DIR, "full_dataset.csv")
    full_df.to_csv(full_path, index=False)

    # Save feature metadata
    features = [
        "num_dependents", "education", "self_employed",
        "income_annum", "loan_amount", "loan_term",
        "cibil_score", "residential_assets", "commercial_assets",
        "luxury_assets", "bank_assets"
    ]
    meta = {
        "features":       features,
        "target":         "default_label",
        "categorical":    ["education", "self_employed"],
        "numeric":        [f for f in features if f not in ["education", "self_employed"]],
        "source_files":   ["loan_approval_dataset.csv", "BankChurners.csv"],
        "drift_schedule": {
            "month_1": "none",
            "month_2": "none",
            "month_3": "data_drift",
            "month_4": "none",
            "month_5": "concept_drift",
            "month_6": "concept_drift"
        }
    }
    with open(os.path.join(DATA_DIR, "feature_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Full dataset: {len(full_df)} rows → {full_path}")
    print(f"  Feature metadata saved.")

    print("\n  Default rate by month:")
    for i in range(1, 7):
        m = full_df[full_df["month"] == i]
        print(f"    Month {i}: {m['default_label'].mean():.2%}  "
              f"({m['drift_type'].iloc[0]})")

    print("\n" + "=" * 60)
    print("  ✅ Real data pipeline complete!")
    print("  Next: python scripts/train_baseline.py")
    print("=" * 60)
