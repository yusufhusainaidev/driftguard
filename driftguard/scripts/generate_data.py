"""
DriftGuard — Synthetic Dataset Generator
Generates 6 months of financial data with injected drift.

Usage:
    python generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_month(n: int, month: int, drift_type: str = None) -> pd.DataFrame:
    # ── Feature distributions ──────────────────────────
    if drift_type == "data_drift":
        amount_mean, amount_std = 85000, 25000
        customer_weights = [0.2, 0.6, 0.2]
    else:
        amount_mean, amount_std = 50000, 15000
        customer_weights = [0.5, 0.3, 0.2]

    transaction_amount    = np.random.normal(amount_mean, amount_std, n).clip(1000, 500000)
    customer_age          = np.random.randint(22, 65, n)
    customer_type         = np.random.choice(["Retail", "Corporate", "SME"], n, p=customer_weights)
    risk_score            = np.random.beta(2, 5, n)
    num_past_defaults     = np.random.poisson(0.3, n)
    transaction_frequency = np.random.randint(1, 30, n)
    region                = np.random.choice(["North", "South", "East", "West"], n)

    # ── Label generation ───────────────────────────────
    if drift_type == "concept_drift":
        # Pattern flips — low risk now defaults
        default_prob = (
            0.6 * (risk_score < 0.3).astype(float) +
            0.05 * (risk_score >= 0.3).astype(float)
        )
    else:
        default_prob = (
            0.7 * risk_score +
            0.1 * (num_past_defaults > 0).astype(float)
        ).clip(0, 1)

    default_label = np.random.binomial(1, default_prob, n)

    start_date = datetime(2024, month, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 28)) for _ in range(n)]

    return pd.DataFrame({
        "date":                   dates,
        "month":                  month,
        "transaction_amount":     transaction_amount.round(2),
        "customer_age":           customer_age,
        "customer_type":          customer_type,
        "risk_score":             risk_score.round(4),
        "num_past_defaults":      num_past_defaults,
        "transaction_frequency":  transaction_frequency,
        "region":                 region,
        "default_label":          default_label,
        "drift_injected":         drift_type if drift_type else "none"
    })


if __name__ == "__main__":
    print("Generating DriftGuard synthetic dataset...")

    months_config = [
        (1000, 1, None),             # Month 1 — clean baseline
        (1000, 2, None),             # Month 2 — clean baseline
        (1000, 3, "data_drift"),     # Month 3 — DATA DRIFT injected
        (1000, 4, None),             # Month 4 — partial recovery
        (1000, 5, "concept_drift"),  # Month 5 — CONCEPT DRIFT injected
        (1000, 6, "concept_drift"),  # Month 6 — still drifted
    ]

    all_months = []
    for n, month, drift_type in months_config:
        df_month = generate_month(n, month, drift_type)
        all_months.append(df_month)
        path = os.path.join(OUTPUT_DIR, f"month_{month}.csv")
        df_month.to_csv(path, index=False)
        print(f"  ✅ Month {month} → {len(df_month)} rows | drift: {drift_type or 'none'} → {path}")

    full_df = pd.concat(all_months, ignore_index=True)
    full_path = os.path.join(OUTPUT_DIR, "full_dataset.csv")
    full_df.to_csv(full_path, index=False)

    print(f"\n✅ Full dataset: {len(full_df)} rows → {full_path}")
    print("\nDrift distribution:")
    print(full_df["drift_injected"].value_counts().to_string())
    print("\nDefault rate by month:")
    print(full_df.groupby("month")["default_label"].mean().round(3).to_string())
    print("\nDone.")
