import os
import json
import redis
import numpy as np
import pandas as pd
from typing import Optional

REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_pass@localhost:6379/0")

# ─────────────────────────────────────────
# Redis Client
# ─────────────────────────────────────────
def get_redis_client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)

# ─────────────────────────────────────────
# Reference Distribution Manager
# Store baseline stats for PSI comparison
# ─────────────────────────────────────────
class ReferenceDistributionCache:

    REFERENCE_KEY = "driftguard:reference_distribution"
    TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

    def __init__(self):
        self.client = get_redis_client()

    def store_reference(self, df: pd.DataFrame) -> bool:
        """
        Compute and store baseline statistics from reference dataframe.
        Called once when baseline (Month 1-2) data is loaded.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        reference_stats = {}

        for col in numeric_cols:
            series = df[col].dropna()
            hist, bin_edges = np.histogram(series, bins=10, density=True)
            reference_stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "histogram": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "count": int(len(series))
            }

        # Store categorical distributions
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            value_counts = df[col].value_counts(normalize=True).to_dict()
            reference_stats[col] = {
                "type": "categorical",
                "distribution": value_counts
            }

        self.client.setex(
            self.REFERENCE_KEY,
            self.TTL_SECONDS,
            json.dumps(reference_stats)
        )
        return True

    def get_reference(self) -> Optional[dict]:
        """Retrieve cached reference distribution."""
        data = self.client.get(self.REFERENCE_KEY)
        if data:
            return json.loads(data)
        return None

    def is_reference_loaded(self) -> bool:
        return self.client.exists(self.REFERENCE_KEY) == 1

    def clear_reference(self) -> bool:
        self.client.delete(self.REFERENCE_KEY)
        return True

    def store_psi_result(self, batch_date: str, psi_scores: dict) -> None:
        """Cache latest PSI scores for dashboard quick access."""
        key = f"driftguard:psi:{batch_date}"
        self.client.setex(key, 60 * 60 * 24, json.dumps(psi_scores))  # 24hr TTL

    def get_psi_result(self, batch_date: str) -> Optional[dict]:
        key = f"driftguard:psi:{batch_date}"
        data = self.client.get(key)
        return json.loads(data) if data else None


# Singleton instance
reference_cache = ReferenceDistributionCache()
