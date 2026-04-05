"""
DriftGuard — Real Data Pipeline
Uses:
  - loan_approval_dataset.csv → Primary training data (baseline)
  - BankChurners.csv          → Drift source (injects realistic distribution shifts)

Produces 6 monthly batches:
  Month 1-2 → Clean baseline (loan_approval data)
  Month 3   → Data Drift (feature distributions shift toward BankChurner stats)
  Month 4   → Partial recovery
  Month 5   → Concept Drift (label relationship flips)
  Month 6   → Still drifted

Usage:
    python scripts/build_data_pipeline.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

LOAN_PATH    = os.path.join(DATASETS_DIR, "loan_approval_dataset.csv")
CHURNER_PATH = os.path.join(DATASETS_DIR, "BankChurners.csv")


# ─────────────────────────────────────────
# STEP 1: Load & Clean Loan Approval Data
# ─────────────────────────────────────────
def load_loan_data() -> pd.DataFrame:
    df = pd.read_csv(LOAN_PATH)
    df.columns = df.columns.str.strip()

    # Clean target
    df["loan_status"] = df["loan_status"].str.strip()
    df["default_label"] = (df["loan_status"] == "Rejected").astype(int)

    # Clean categoricals
    df["education"]     = df["education"].str.strip()
    df["self_employed"] = df["self_employed"].str.strip()

    # Normalize income & loan amounts (scale down for readability)
    df["income_annum"]  = df["income_annum"] / 100000   # in lakhs
    df["loan_amount"]   = df["loan_amount"]  / 100000

    # Drop original status col
    df = df.drop(columns=["loan_id", "loan_status"])

    print(f"  Loan data loaded: {len(df)} rows")
    print(f"  Approval rate : {(df['default_label']==0).mean():.2%}")
    print(f"  Rejection rate: {(df['default_label']==1).mean():.2%}")
    return df


# ─────────────────────────────────────────
# STEP 2: Extract BankChurner Drift Stats
# ─────────────────────────────────────────
def extract_churner_drift_stats() -> dict:
    df = pd.read_csv(CHURNER_PATH)

    # Map BankChurner stats → loan approval feature space
    # We use churner stats to shift means/stds of comparable features
    drift_stats = {
        "income_mean_shift":      df["Credit_Limit"].mean() / 100000,     # credit limit as proxy for income
        "income_std_shift":       df["Credit_Limit"].std()  / 100000,
        "loan_mean_shift":        df["Total_Trans_Amt"].mean() / 10000,
        "cibil_mean_shift":       300 + (df["Avg_Utilization_Ratio"].mean() * 400),  # map 0-1 util → 300-700 CIBIL range
        "cibil_std_shift":        df["Avg_Utilization_Ratio"].std() * 200,
        "self_employed_rate":     df[df["Income_Category"] == "$120K +"].shape[0] / len(df),
        "high_education_rate":    df[df["Education_Level"] == "Graduate"].shape[0] / len(df),
    }

    print(f"\n  BankChurner drift stats extracted:")
    for k, v in drift_stats.items():
        print(f"    {k:30s}: {v:.4f}")

    return drift_stats


# ─────────────────────────────────────────
# STEP 3: Build Monthly Batches
# ─────────────────────────────────────────
def build_monthly_batches(df: pd.DataFrame, drift_stats: dict):

    n_total = len(df)
    batch_size = n_total // 6  # ~711 rows per month

    # Shuffle baseline data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    months = []

    for month in range(1, 7):
        start = (month - 1) * batch_size
        end   = start + batch_size if month < 6 else n_total
        batch = df_shuffled.iloc[start:end].copy()
        batch["month"] = month
        batch["drift_injected"] = "none"

        # ── Month 3: DATA DRIFT ────────────────────────────────────
        if month == 3:
            batch = inject_data_drift(batch, drift_stats)
            batch["drift_injected"] = "data_drift"

        # ── Month 4: Partial recovery (mild drift) ─────────────────
        elif month == 4:
            batch = inject_data_drift(batch, drift_stats, intensity=0.4)
            batch["drift_injected"] = "mild_drift"

        # ── Month 5: CONCEPT DRIFT ─────────────────────────────────
        elif month == 5:
            batch = inject_concept_drift(batch)
            batch["drift_injected"] = "concept_drift"

        # ── Month 6: Still concept drifted ─────────────────────────
        elif month == 6:
            batch = inject_concept_drift(batch, intensity=0.8)
            batch["drift_injected"] = "concept_drift"

        months.append(batch)
        print(f"  Month {month}: {len(batch)} rows | drift: {batch['drift_injected'].iloc[0]}")

    return months


# ─────────────────────────────────────────
# STEP 4: Inject Data Drift
# Shifts feature distributions (not labels)
# ─────────────────────────────────────────
def inject_data_drift(batch: pd.DataFrame, drift_stats: dict, intensity: float = 1.0) -> pd.DataFrame:
    batch = batch.copy()
    n = len(batch)

    # Shift income distribution (BankChurner credit limit stats)
    income_noise = np.random.normal(
        drift_stats["income_mean_shift"] * intensity,
        drift_stats["income_std_shift"] * 0.3,
        n
    )
    batch["income_annum"] = (batch["income_annum"] + income_noise).clip(1, None)

    # Shift CIBIL score distribution
    cibil_shift = (drift_stats["cibil_mean_shift"] - batch["cibil_score"].mean()) * intensity
    cibil_noise = np.random.normal(cibil_shift, drift_stats["cibil_std_shift"] * 0.2, n)
    batch["cibil_score"] = (batch["cibil_score"] + cibil_noise).clip(300, 900).astype(int)

    # Shift loan amount
    loan_noise = np.random.normal(
        drift_stats["loan_mean_shift"] * intensity,
        drift_stats["loan_mean_shift"] * 0.2,
        n
    )
    batch["loan_amount"] = (batch["loan_amount"] + loan_noise).clip(1, None)

    # Shift self_employed rate
    n_flip = int(n * drift_stats["self_employed_rate"] * intensity * 0.3)
    flip_idx = batch.sample(n=min(n_flip, n), random_state=42).index
    batch.loc[flip_idx, "self_employed"] = "Yes"

    return batch


# ─────────────────────────────────────────
# STEP 5: Inject Concept Drift
# Flips label relationship with features
# ─────────────────────────────────────────
def inject_concept_drift(batch: pd.DataFrame, intensity: float = 1.0) -> pd.DataFrame:
    batch = batch.copy()
    n = len(batch)

    # Flip: High CIBIL score customers now get REJECTED (pattern reversal)
    # In real world: bank policy change, economic crisis, new credit norms
    high_cibil_mask = batch["cibil_score"] > 700
    n_flip = int(high_cibil_mask.sum() * 0.6 * intensity)

    if n_flip > 0:
        flip_idx = batch[high_cibil_mask].sample(
            n=min(n_flip, high_cibil_mask.sum()),
            random_state=42
        ).index
        batch.loc[flip_idx, "default_label"] = 1  # approved customers now rejected

    # Flip: Low income customers now get APPROVED
    low_income_mask = batch["income_annum"] < batch["income_annum"].quantile(0.3)
    n_flip2 = int(low_income_mask.sum() * 0.5 * intensity)

    if n_flip2 > 0:
        flip_idx2 = batch[low_income_mask].sample(
            n=min(n_flip2, low_income_mask.sum()),
            random_state=42
        ).index
        batch.loc[flip_idx2, "default_label"] = 0

    return batch


# ─────────────────────────────────────────
# STEP 6: Save Everything
# ─────────────────────────────────────────
def save_batches(months: list):
    all_data = pd.concat(months, ignore_index=True)

    # Full dataset
    full_path = os.path.join(DATA_DIR, "full_dataset.csv")
    all_data.to_csv(full_path, index=False)
    print(f"\n  Full dataset → {full_path} ({len(all_data)} rows)")

    # Per-month files
    for batch in months:
        m = batch["month"].iloc[0]
        path = os.path.join(DATA_DIR, f"month_{m}.csv")
        batch.to_csv(path, index=False)
        print(f"  month_{m}.csv → {len(batch)} rows | rejection rate: {batch['default_label'].mean():.2%}")

    # Save feature list + metadata
    feature_cols = [c for c in all_data.columns
                    if c not in ["default_label", "month", "drift_injected"]]

    metadata = {
        "features":          feature_cols,
        "target":            "default_label",
        "total_rows":        len(all_data),
        "months":            6,
        "source_primary":    "loan_approval_dataset.csv",
        "source_drift":      "BankChurners.csv",
        "drift_schedule": {
            "month_1": "baseline",
            "month_2": "baseline",
            "month_3": "data_drift",
            "month_4": "mild_drift",
            "month_5": "concept_drift",
            "month_6": "concept_drift"
        }
    }

    meta_path = os.path.join(DATA_DIR, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata      → {meta_path}")

    return all_data, metadata


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DriftGuard — Real Data Pipeline Builder")
    print("=" * 55)

    print("\n[1/4] Loading loan approval data...")
    df = load_loan_data()

    print("\n[2/4] Extracting BankChurner drift statistics...")
    drift_stats = extract_churner_drift_stats()

    print("\n[3/4] Building 6 monthly batches with drift injection...")
    months = build_monthly_batches(df, drift_stats)

    print("\n[4/4] Saving all files...")
    all_data, metadata = save_batches(months)

    print("\n" + "=" * 55)
    print("  ✅ Data pipeline complete!")
    print(f"  Total rows      : {len(all_data)}")
    print(f"  Features        : {len(metadata['features'])}")
    print(f"  Rejection rates by month:")
    for batch in months:
        m = batch['month'].iloc[0]
        print(f"    Month {m}: {batch['default_label'].mean():.2%} ({batch['drift_injected'].iloc[0]})")
    print("=" * 55)
    print("\n  ⚠️  Copy loan_approval_dataset.csv and BankChurners.csv")
    print("      into the datasets/ folder before running this script.")
