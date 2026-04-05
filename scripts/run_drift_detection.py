"""
DriftGuard — Step 3: Drift Detection Module
Detects drift using PSI + KS Test + ADWIN
Reads reference from Redis, compares incoming batches

Usage:
    python scripts/run_drift_detection.py --month 3
    python scripts/run_drift_detection.py --month 5
    python scripts/run_drift_detection.py --all
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────
PSI_THRESHOLD         = float(os.getenv("PSI_THRESHOLD", 0.2))
KS_THRESHOLD          = float(os.getenv("KS_THRESHOLD", 0.05))
ACCURACY_DROP_THRESH  = float(os.getenv("ACCURACY_DROP_THRESHOLD", 0.05))

# PSI severity levels
PSI_LOW      = 0.1   # minor shift
PSI_MEDIUM   = 0.2   # moderate drift — monitor
PSI_HIGH     = 0.25  # significant drift — alert


# ═══════════════════════════════════════════════════════
# 1. PSI — Population Stability Index
#    Detects DATA DRIFT (distribution shift in features)
# ═══════════════════════════════════════════════════════
class PSIDetector:
    """
    PSI measures how much a feature's distribution has
    shifted compared to the reference (baseline) distribution.

    PSI < 0.1  → No significant change
    PSI 0.1-0.2 → Moderate change, monitor
    PSI > 0.2  → Significant drift, action needed
    """

    def __init__(self, n_bins: int = 10, eps: float = 1e-6):
        self.n_bins = n_bins
        self.eps    = eps

    def compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Compute PSI between reference and current distribution."""
        # Build bins from reference
        breakpoints = np.percentile(reference, np.linspace(0, 100, self.n_bins + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 3:
            return 0.0  # Not enough distinct values

        # Count proportions in each bin
        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current,   bins=breakpoints)

        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)

        # Avoid log(0)
        ref_pct = np.where(ref_pct == 0, self.eps, ref_pct)
        cur_pct = np.where(cur_pct == 0, self.eps, cur_pct)

        # PSI formula: sum((current% - reference%) * ln(current% / reference%))
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def compute_all_features(self, ref_df: pd.DataFrame,
                              cur_df: pd.DataFrame,
                              features: list) -> dict:
        """Compute PSI for all numeric features."""
        results = {}
        for feat in features:
            if feat not in ref_df.columns or feat not in cur_df.columns:
                continue
            if ref_df[feat].dtype not in [np.float64, np.int64, float, int]:
                continue
            psi_val = self.compute_psi(
                ref_df[feat].dropna().values,
                cur_df[feat].dropna().values
            )
            severity = self._get_severity(psi_val)
            results[feat] = {
                "psi":       round(psi_val, 4),
                "severity":  severity,
                "drifted":   psi_val > PSI_THRESHOLD
            }
        return results

    def _get_severity(self, psi: float) -> str:
        if psi < PSI_LOW:     return "none"
        if psi < PSI_MEDIUM:  return "low"
        if psi < PSI_HIGH:    return "medium"
        return "high"


# ═══════════════════════════════════════════════════════
# 2. KS Test — Kolmogorov-Smirnov Test
#    Statistical test for distribution difference
#    More sensitive than PSI for subtle shifts
# ═══════════════════════════════════════════════════════
class KSDetector:
    """
    KS Test checks whether two samples come from the
    same distribution using the KS statistic.

    p-value < 0.05 → Distributions are significantly different
    """

    def compute_ks(self, reference: np.ndarray,
                   current: np.ndarray) -> tuple:
        """Returns (ks_statistic, p_value)"""
        ks_stat, p_value = stats.ks_2samp(reference, current)
        return float(ks_stat), float(p_value)

    def compute_all_features(self, ref_df: pd.DataFrame,
                              cur_df: pd.DataFrame,
                              features: list) -> dict:
        results = {}
        for feat in features:
            if feat not in ref_df.columns or feat not in cur_df.columns:
                continue
            if ref_df[feat].dtype not in [np.float64, np.int64, float, int]:
                continue
            ks_stat, p_val = self.compute_ks(
                ref_df[feat].dropna().values,
                cur_df[feat].dropna().values
            )
            results[feat] = {
                "ks_statistic": round(ks_stat, 4),
                "p_value":      round(p_val, 6),
                "drifted":      p_val < KS_THRESHOLD
            }
        return results


# ═══════════════════════════════════════════════════════
# 3. ADWIN — Adaptive Windowing
#    Real-time drift detector on prediction error stream
#    Detects CONCEPT DRIFT by monitoring accuracy
# ═══════════════════════════════════════════════════════
class ADWINDetector:
    """
    ADWIN maintains a sliding window of model errors.
    When it detects a statistical change in error rate,
    it signals concept drift.

    This is our real-time detector — it works on the
    stream of correct/incorrect predictions.
    """

    def __init__(self, delta: float = 0.002):
        self.delta       = delta  # confidence parameter
        self.window      = []     # sliding window of errors (0/1)
        self.drift_detected = False
        self.drift_at    = None

    def add_element(self, error: int) -> bool:
        """
        Add a prediction error (1=wrong, 0=correct).
        Returns True if drift detected.
        """
        self.window.append(error)
        self.drift_detected = self._check_drift()
        return self.drift_detected

    def _check_drift(self) -> bool:
        """
        ADWIN algorithm: check if any split of the window
        shows statistically different error rates.
        """
        n = len(self.window)
        if n < 30:  # need minimum samples
            return False

        total_errors = sum(self.window)
        total_mean   = total_errors / n

        # Check all possible split points
        for split in range(10, n - 10):
            w0 = self.window[:split]
            w1 = self.window[split:]

            n0    = len(w0)
            n1    = len(w1)
            mean0 = sum(w0) / n0
            mean1 = sum(w1) / n1

            # Hoeffding bound
            epsilon_cut = self._compute_epsilon(n0, n1)

            if abs(mean0 - mean1) >= epsilon_cut:
                # Drift detected — drop older part of window
                self.window  = self.window[split:]
                self.drift_at = len(self.window)
                return True

        return False

    def _compute_epsilon(self, n0: int, n1: int) -> float:
        """Compute Hoeffding bound threshold."""
        n    = n0 + n1
        m    = 1 / (1/n0 + 1/n1) if (n0 > 0 and n1 > 0) else 1
        return float(np.sqrt((1 / (2 * m)) * np.log(4 * n / self.delta)))

    def reset(self):
        self.window = []
        self.drift_detected = False

    @property
    def error_rate(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)


# ═══════════════════════════════════════════════════════
# 4. Concept Drift Detector
#    Combines model accuracy drop + ADWIN
# ═══════════════════════════════════════════════════════
class ConceptDriftDetector:
    """
    Detects concept drift by:
    1. Measuring accuracy drop vs baseline
    2. Running ADWIN on prediction error stream
    """

    def __init__(self, baseline_accuracy: float):
        self.baseline_accuracy = baseline_accuracy
        self.adwin = ADWINDetector(delta=0.002)

    def check_accuracy_drop(self, current_accuracy: float) -> dict:
        drop = self.baseline_accuracy - current_accuracy
        return {
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "current_accuracy":  round(current_accuracy, 4),
            "accuracy_drop":     round(drop, 4),
            "drifted":           drop > ACCURACY_DROP_THRESH,
            "severity":          self._severity(drop)
        }

    def run_adwin_on_batch(self, y_true: np.ndarray,
                           y_pred: np.ndarray) -> dict:
        """Feed prediction errors into ADWIN stream."""
        self.adwin.reset()
        drift_points = []

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            error = int(true != pred)
            if self.adwin.add_element(error):
                drift_points.append(i)

        return {
            "adwin_drift_detected": self.adwin.drift_detected,
            "drift_points":         drift_points,
            "final_error_rate":     round(self.adwin.error_rate, 4),
            "window_size":          len(self.adwin.window)
        }

    def _severity(self, drop: float) -> str:
        if drop < 0.02:   return "none"
        if drop < 0.05:   return "low"
        if drop < 0.10:   return "medium"
        return "high"


# ═══════════════════════════════════════════════════════
# 5. Main Drift Runner
# ═══════════════════════════════════════════════════════
class DriftRunner:

    def __init__(self):
        self.psi_detector = PSIDetector()
        self.ks_detector  = KSDetector()
        self._load_baseline_model()
        self._load_reference_data()

    def _load_baseline_model(self):
        """Load trained XGBoost model and encoders."""
        model_path = os.path.join(MODEL_DIR, "model_v1.0.0.pkl")
        enc_path   = os.path.join(MODEL_DIR, "encoders.pkl")
        meta_path  = os.path.join(MODEL_DIR, "metadata_v1.0.0.json")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(enc_path, "rb") as f:
            self.encoders = pickle.load(f)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.features          = self.metadata["features"]
        self.baseline_accuracy = self.metadata["metrics"]["accuracy"]
        self.baseline_auc      = self.metadata["metrics"]["auc_roc"]

        print(f"  Model loaded: v1.0.0")
        print(f"  Baseline accuracy: {self.baseline_accuracy}")
        print(f"  Features: {self.features}")

    def _load_reference_data(self):
        """Load Month 1 + 2 as reference distribution."""
        dfs = []
        for m in [1, 2]:
            path = os.path.join(DATA_DIR, f"month_{m}.csv")
            dfs.append(pd.read_csv(path))
        self.reference_df = pd.concat(dfs, ignore_index=True)
        print(f"  Reference data: {len(self.reference_df)} rows (Month 1 & 2)")

    def _preprocess_batch(self, df: pd.DataFrame) -> tuple:
        """Preprocess incoming batch using saved encoders."""
        df = df.copy()
        drop_cols = ["date", "month", "drift_type", "default_label",
                     "loan_id", "loan_status"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns
                               and c != "default_label"])

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        X = df[[f for f in self.features if f in df.columns]]
        y = df["default_label"] if "default_label" in df.columns else None
        return X, y

    def run_for_month(self, month: int) -> dict:
        """Run full drift detection for a given month."""
        print(f"\n{'='*60}")
        print(f"  Running Drift Detection — Month {month}")
        print(f"{'='*60}")

        # Load batch
        batch_path = os.path.join(DATA_DIR, f"month_{month}.csv")
        batch_df   = pd.read_csv(batch_path)
        actual_drift = batch_df["drift_type"].iloc[0] if "drift_type" in batch_df.columns else "unknown"
        print(f"\n  Batch: {len(batch_df)} rows | actual drift: {actual_drift}")

        # Numeric features only for distribution checks
        numeric_features = [f for f in self.features
                            if f not in ["education", "self_employed"]]

        # ── PSI Detection ──────────────────────────────
        print(f"\n[PSI] Computing Population Stability Index...")
        psi_results = self.psi_detector.compute_all_features(
            self.reference_df, batch_df, numeric_features
        )
        drifted_features_psi = [f for f, r in psi_results.items() if r["drifted"]]
        max_psi = max((r["psi"] for r in psi_results.values()), default=0)

        print(f"  PSI scores:")
        for feat, res in sorted(psi_results.items(), key=lambda x: x[1]["psi"], reverse=True):
            flag = "⚠️  DRIFT" if res["drifted"] else "✅"
            print(f"    {feat:28s}: {res['psi']:.4f}  [{res['severity']}]  {flag}")
        print(f"  Drifted features (PSI): {drifted_features_psi}")

        # ── KS Test ────────────────────────────────────
        print(f"\n[KS ] Running Kolmogorov-Smirnov Test...")
        ks_results = self.ks_detector.compute_all_features(
            self.reference_df, batch_df, numeric_features
        )
        drifted_features_ks = [f for f, r in ks_results.items() if r["drifted"]]

        print(f"  KS results:")
        for feat, res in sorted(ks_results.items(), key=lambda x: x[1]["ks_statistic"], reverse=True):
            flag = "⚠️  DRIFT" if res["drifted"] else "✅"
            print(f"    {feat:28s}: stat={res['ks_statistic']:.4f}  p={res['p_value']:.4f}  {flag}")
        print(f"  Drifted features (KS): {drifted_features_ks}")

        # ── Accuracy & ADWIN ───────────────────────────
        print(f"\n[ADWIN] Running accuracy + ADWIN check...")
        X_batch, y_batch = self._preprocess_batch(batch_df)

        concept_result = {"accuracy_check": {}, "adwin_result": {}}
        if y_batch is not None:
            y_pred  = self.model.predict(X_batch)
            current_acc = float((y_pred == y_batch.values).mean())

            concept_detector = ConceptDriftDetector(self.baseline_accuracy)
            acc_result  = concept_detector.check_accuracy_drop(current_acc)
            adwin_result = concept_detector.run_adwin_on_batch(y_batch.values, y_pred)

            concept_result = {
                "accuracy_check": acc_result,
                "adwin_result":   adwin_result
            }

            print(f"  Accuracy: baseline={self.baseline_accuracy:.4f}  "
                  f"current={current_acc:.4f}  "
                  f"drop={acc_result['accuracy_drop']:.4f}")
            print(f"  ADWIN drift detected: {adwin_result['adwin_drift_detected']}")
            print(f"  Error rate: {adwin_result['final_error_rate']:.4f}")

        # ── Final Verdict ──────────────────────────────
        data_drift_detected    = len(drifted_features_psi) >= 2 or max_psi > PSI_THRESHOLD
        concept_drift_detected = (
            concept_result.get("accuracy_check", {}).get("drifted", False) or
            concept_result.get("adwin_result", {}).get("adwin_drift_detected", False)
        )
        any_drift = data_drift_detected or concept_drift_detected

        print(f"\n{'─'*60}")
        print(f"  DRIFT VERDICT — Month {month}")
        print(f"  Data Drift    : {'⚠️  DETECTED' if data_drift_detected else '✅ None'}")
        print(f"  Concept Drift : {'⚠️  DETECTED' if concept_drift_detected else '✅ None'}")
        print(f"  Action Needed : {'🔴 YES — Trigger Retraining' if any_drift else '🟢 NO'}")
        print(f"{'─'*60}")

        # ── Build Result ───────────────────────────────
        result = {
            "month":                    month,
            "batch_date":               f"2024-{month:02d}-01",
            "batch_size":               len(batch_df),
            "actual_drift_injected":    actual_drift,
            "psi_results":              psi_results,
            "ks_results":               ks_results,
            "concept_drift":            concept_result,
            "drifted_features_psi":     drifted_features_psi,
            "drifted_features_ks":      drifted_features_ks,
            "max_psi":                  round(max_psi, 4),
            "data_drift_detected":      data_drift_detected,
            "concept_drift_detected":   concept_drift_detected,
            "any_drift_detected":       any_drift,
            "trigger_retrain":          any_drift,
            "detected_at":              datetime.now().isoformat()
        }

        # Save result to logs
        log_path = os.path.join(LOGS_DIR, f"drift_month_{month}.json")
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Result saved → {log_path}")

        return result


# ─────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriftGuard — Drift Detection")
    parser.add_argument("--month", type=int, help="Month number (1-6)")
    parser.add_argument("--all",   action="store_true", help="Run all months")
    args = parser.parse_args()

    runner = DriftRunner()

    if args.all:
        all_results = []
        for m in range(1, 7):
            r = runner.run_for_month(m)
            all_results.append(r)

        # Summary table
        print(f"\n{'═'*60}")
        print(f"  SUMMARY — All Months")
        print(f"{'═'*60}")
        print(f"  {'Month':<8} {'Actual':<15} {'Data Drift':<14} {'Concept':<12} {'Retrain?'}")
        print(f"  {'─'*55}")
        for r in all_results:
            print(f"  {r['month']:<8} {r['actual_drift_injected']:<15} "
                  f"{'YES' if r['data_drift_detected'] else 'no':<14} "
                  f"{'YES' if r['concept_drift_detected'] else 'no':<12} "
                  f"{'🔴 YES' if r['trigger_retrain'] else '🟢 no'}")

        summary_path = os.path.join(LOGS_DIR, "drift_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Full summary saved → {summary_path}")

    elif args.month:
        runner.run_for_month(args.month)
    else:
        print("Usage:")
        print("  python scripts/run_drift_detection.py --month 3")
        print("  python scripts/run_drift_detection.py --all")
