"""
DriftGuard — Step 4: Decision Engine
Reads drift detection results and decides:
  1. Should we retrain?
  2. If retrained, should we deploy the new model?
  3. Or do we rollback?

This is the brain of DriftGuard.

Usage:
    python scripts/decision_engine.py --month 3
    python scripts/decision_engine.py --all
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
IMPROVEMENT_MARGIN   = float(os.getenv("RETRAIN_IMPROVEMENT_MARGIN", 0.01))  # 1% improvement needed
MIN_RETRAIN_SAMPLES  = int(os.getenv("MIN_RETRAIN_SAMPLES", 300))
RETRAIN_WINDOW_MONTHS = 2   # use last N months for retraining

XGBOOST_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1
}

DROP_COLS = ["date", "month", "drift_type", "loan_id", "loan_status"]
TARGET    = "default_label"


# ═══════════════════════════════════════════════════════
# Decision Rules — The Core Logic
# ═══════════════════════════════════════════════════════
class DecisionRules:
    """
    Clear, explainable rules for retraining decision.
    This makes the system auditable — important for finance.
    """

    @staticmethod
    def should_retrain(drift_result: dict) -> tuple:
        """
        Returns (should_retrain: bool, reason: str)

        Rules (in priority order):
        1. Concept drift detected → ALWAYS retrain
        2. Data drift on >2 features AND accuracy drop → retrain
        3. Data drift on >3 features → retrain (preventive)
        4. Max PSI > 0.25 → retrain
        5. Otherwise → no retrain
        """
        concept_drift = drift_result.get("concept_drift_detected", False)
        data_drift    = drift_result.get("data_drift_detected", False)
        n_drifted     = len(drift_result.get("drifted_features_psi", []))
        max_psi       = drift_result.get("max_psi", 0)
        acc_check     = drift_result.get("concept_drift", {}).get("accuracy_check", {})
        acc_drop      = acc_check.get("accuracy_drop", 0)

        # Rule 1
        if concept_drift:
            return True, f"Concept drift detected (accuracy drop={acc_drop:.3f}, ADWIN triggered)"

        # Rule 2
        if data_drift and acc_drop > 0.02:
            return True, f"Data drift ({n_drifted} features) + accuracy drop ({acc_drop:.3f})"

        # Rule 3
        if n_drifted >= 3:
            return True, f"Preventive retrain: {n_drifted} features drifted"

        # Rule 4
        if max_psi > 0.25:
            return True, f"Critical PSI score: {max_psi:.3f} (threshold=0.25)"

        return False, "No significant drift detected — model stable"

    @staticmethod
    def should_deploy(old_metrics: dict, new_metrics: dict) -> tuple:
        """
        Returns (should_deploy: bool, reason: str)

        Deploy only if new model is meaningfully better.
        In finance, we require improvement on MULTIPLE metrics
        to avoid deploying a model that overfits to recent drift.
        """
        old_auc = old_metrics.get("auc_roc", 0)
        new_auc = new_metrics.get("auc_roc", 0)
        old_f1  = old_metrics.get("f1_score", 0)
        new_f1  = new_metrics.get("f1_score", 0)
        old_acc = old_metrics.get("accuracy", 0)
        new_acc = new_metrics.get("accuracy", 0)

        auc_better = new_auc  > old_auc + IMPROVEMENT_MARGIN
        f1_better  = new_f1   > old_f1  + IMPROVEMENT_MARGIN
        acc_better = new_acc  > old_acc

        # Deploy if AUC improves meaningfully AND at least one other metric improves
        if auc_better and (f1_better or acc_better):
            return True, (
                f"New model better: AUC {old_auc:.4f}→{new_auc:.4f}, "
                f"F1 {old_f1:.4f}→{new_f1:.4f}"
            )

        # Deploy if all metrics improve even slightly
        if new_auc >= old_auc and new_f1 >= old_f1 and new_acc >= old_acc:
            return True, (
                f"New model marginally better across all metrics: "
                f"AUC {old_auc:.4f}→{new_auc:.4f}"
            )

        return False, (
            f"New model NOT better enough: "
            f"AUC {old_auc:.4f}→{new_auc:.4f}, "
            f"F1 {old_f1:.4f}→{new_f1:.4f} "
            f"(required improvement: {IMPROVEMENT_MARGIN})"
        )


# ═══════════════════════════════════════════════════════
# Retrainer — Trains new model on recent data
# ═══════════════════════════════════════════════════════
class Retrainer:

    def __init__(self):
        with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
            self.encoders = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "metadata_v1.0.0.json"), "r") as f:
            self.base_metadata = json.load(f)
        self.features = self.base_metadata["features"]

    def _preprocess(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))

        X = df[[f for f in self.features if f in df.columns]]
        y = df[TARGET]
        return X, y

    def retrain(self, trigger_month: int, new_version: str) -> dict:
        """
        Retrain on a window of recent data.
        Uses baseline months + months up to trigger point.
        """
        print(f"\n  Retraining on months up to {trigger_month}...")

        # Collect training data — baseline + recent months
        train_months = list(range(1, trigger_month + 1))
        dfs = []
        for m in train_months:
            path = os.path.join(DATA_DIR, f"month_{m}.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))

        combined = pd.concat(dfs, ignore_index=True)
        print(f"  Training data: {len(combined)} rows from months {train_months}")

        if len(combined) < MIN_RETRAIN_SAMPLES:
            return {"error": f"Not enough samples: {len(combined)} < {MIN_RETRAIN_SAMPLES}"}

        X, y = self._preprocess(combined)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train new model
        new_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        new_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate
        y_pred  = new_model.predict(X_test)
        y_proba = new_model.predict_proba(X_test)[:, 1]

        new_metrics = {
            "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc_roc":   round(float(roc_auc_score(y_test, y_proba)), 4),
        }

        print(f"  New model metrics: {new_metrics}")

        # Save new model
        model_path = os.path.join(MODEL_DIR, f"model_{new_version}.pkl")
        meta_path  = os.path.join(MODEL_DIR, f"metadata_{new_version}.json")

        with open(model_path, "wb") as f:
            pickle.dump(new_model, f)

        metadata = {
            "version":          new_version,
            "features":         self.features,
            "target":           TARGET,
            "metrics":          new_metrics,
            "training_months":  train_months,
            "xgboost_params":   XGBOOST_PARAMS,
            "status":           "candidate",
            "trained_at":       datetime.now().isoformat()
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "new_version":   new_version,
            "new_metrics":   new_metrics,
            "model_path":    model_path,
            "training_rows": len(combined),
            "model":         new_model
        }


# ═══════════════════════════════════════════════════════
# Rollback Manager
# ═══════════════════════════════════════════════════════
class RollbackManager:

    @staticmethod
    def get_active_model_version() -> str:
        """Find the currently active model version."""
        for fname in sorted(os.listdir(MODEL_DIR), reverse=True):
            if fname.startswith("metadata_") and fname.endswith(".json"):
                path = os.path.join(MODEL_DIR, fname)
                with open(path) as f:
                    meta = json.load(f)
                if meta.get("status") == "active":
                    return meta["version"]
        return "v1.0.0"

    @staticmethod
    def activate_model(version: str) -> None:
        """Set a model as active, retire all others."""
        for fname in os.listdir(MODEL_DIR):
            if fname.startswith("metadata_") and fname.endswith(".json"):
                path = os.path.join(MODEL_DIR, fname)
                with open(path) as f:
                    meta = json.load(f)
                if meta["version"] == version:
                    meta["status"] = "active"
                elif meta.get("status") == "active":
                    meta["status"] = "retired"
                with open(path, "w") as f:
                    json.dump(meta, f, indent=2)

    @staticmethod
    def rollback_to(version: str) -> dict:
        """Rollback to a specific model version."""
        meta_path  = os.path.join(MODEL_DIR, f"metadata_{version}.json")
        model_path = os.path.join(MODEL_DIR, f"model_{version}.pkl")

        if not os.path.exists(model_path):
            return {"success": False, "error": f"Model {version} not found"}

        RollbackManager.activate_model(version)
        return {
            "success":          True,
            "rolled_back_to":   version,
            "timestamp":        datetime.now().isoformat()
        }


# ═══════════════════════════════════════════════════════
# Main Decision Engine
# ═══════════════════════════════════════════════════════
class DecisionEngine:

    def __init__(self):
        self.rules    = DecisionRules()
        self.retrainer = Retrainer()
        self.rollback  = RollbackManager()

        # Load active model metrics
        active_version = self.rollback.get_active_model_version()
        meta_path = os.path.join(MODEL_DIR, f"metadata_{active_version}.json")
        with open(meta_path) as f:
            self.active_metadata = json.load(f)
        self.active_metrics  = self.active_metadata["metrics"]
        self.active_version  = active_version
        print(f"  Active model: {active_version}")
        print(f"  Active metrics: {self.active_metrics}")

    def run_for_month(self, month: int) -> dict:
        """Full decision pipeline for a given month."""
        print(f"\n{'═'*60}")
        print(f"  Decision Engine — Month {month}")
        print(f"{'═'*60}")

        # Load drift result
        drift_log = os.path.join(LOGS_DIR, f"drift_month_{month}.json")
        if not os.path.exists(drift_log):
            print(f"  ⚠️  No drift result for month {month}.")
            print(f"      Run: python scripts/run_drift_detection.py --month {month}")
            return {}

        with open(drift_log) as f:
            drift_result = json.load(f)

        # ── Step 1: Should we retrain? ─────────────────
        print(f"\n[DECISION 1] Should we retrain?")
        should_retrain, retrain_reason = self.rules.should_retrain(drift_result)
        print(f"  Decision : {'🔴 YES' if should_retrain else '🟢 NO'}")
        print(f"  Reason   : {retrain_reason}")

        decision_log = {
            "month":           month,
            "active_model":    self.active_version,
            "drift_detected":  drift_result.get("any_drift_detected", False),
            "retrain_decision": should_retrain,
            "retrain_reason":  retrain_reason,
            "deploy_decision": None,
            "deploy_reason":   None,
            "new_version":     None,
            "outcome":         None,
            "timestamp":       datetime.now().isoformat()
        }

        if not should_retrain:
            decision_log["outcome"] = "NO_ACTION"
            self._save_decision(decision_log, month)
            print(f"\n  Outcome: 🟢 NO ACTION — Model remains {self.active_version}")
            return decision_log

        # ── Step 2: Retrain ────────────────────────────
        print(f"\n[DECISION 2] Retraining model...")
        # e.g. v1.0.0 → major=1 → new = v2.0.0
        major       = int(self.active_version.lstrip("v").split(".")[0])
        new_version = f"v{major + 1}.0.0"
        retrain_result = self.retrainer.retrain(month, new_version)

        if "error" in retrain_result:
            decision_log["outcome"] = "RETRAIN_FAILED"
            decision_log["error"]   = retrain_result["error"]
            self._save_decision(decision_log, month)
            print(f"  ❌ Retrain failed: {retrain_result['error']}")
            return decision_log

        new_metrics = retrain_result["new_metrics"]
        decision_log["new_version"] = new_version

        print(f"\n  Old model ({self.active_version}): {self.active_metrics}")
        print(f"  New model ({new_version}):           {new_metrics}")

        # ── Step 3: A/B Test — Deploy or Reject? ──────
        print(f"\n[DECISION 3] A/B Test — Deploy new model?")
        should_deploy, deploy_reason = self.rules.should_deploy(
            self.active_metrics, new_metrics
        )
        print(f"  Decision : {'✅ DEPLOY' if should_deploy else '❌ REJECT'}")
        print(f"  Reason   : {deploy_reason}")

        decision_log["deploy_decision"] = should_deploy
        decision_log["deploy_reason"]   = deploy_reason

        if should_deploy:
            self.rollback.activate_model(new_version)
            self.active_version = new_version
            self.active_metrics = new_metrics
            decision_log["outcome"] = "DEPLOYED"
            print(f"\n  Outcome: ✅ DEPLOYED — Active model is now {new_version}")
        else:
            # Mark new model as rejected
            meta_path = os.path.join(MODEL_DIR, f"metadata_{new_version}.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["status"] = "rejected"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

            decision_log["outcome"] = "REJECTED"
            print(f"\n  Outcome: ❌ REJECTED — Keeping {self.active_version}")
            print(f"           Old model retained. Consider rollback if needed.")

        self._save_decision(decision_log, month)
        self._print_summary(decision_log)
        return decision_log

    def _save_decision(self, log: dict, month: int) -> None:
        path = os.path.join(LOGS_DIR, f"decision_month_{month}.json")
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n  Decision log saved → {path}")

    def _print_summary(self, log: dict) -> None:
        print(f"\n{'─'*60}")
        print(f"  FINAL SUMMARY")
        print(f"{'─'*60}")
        print(f"  Month          : {log['month']}")
        print(f"  Drift Found    : {log['drift_detected']}")
        print(f"  Retrain        : {log['retrain_decision']} — {log['retrain_reason']}")
        print(f"  Deploy         : {log['deploy_decision']} — {log.get('deploy_reason','')}")
        print(f"  Active Model   : {log.get('new_version') if log['outcome']=='DEPLOYED' else log['active_model']}")
        print(f"  Outcome        : {log['outcome']}")
        print(f"{'─'*60}")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriftGuard — Decision Engine")
    parser.add_argument("--month", type=int,  help="Run for specific month")
    parser.add_argument("--all",   action="store_true", help="Run all months")
    parser.add_argument("--rollback", type=str, help="Rollback to version (e.g. v1.0.0)")
    args = parser.parse_args()

    if args.rollback:
        manager = RollbackManager()
        result  = manager.rollback_to(args.rollback)
        print(f"Rollback result: {result}")
        sys.exit(0)

    engine = DecisionEngine()

    if args.all:
        all_decisions = []
        for m in range(1, 7):
            drift_log = os.path.join(LOGS_DIR, f"drift_month_{m}.json")
            if not os.path.exists(drift_log):
                print(f"  Skipping month {m} — no drift result. Run drift detection first.")
                continue
            d = engine.run_for_month(m)
            all_decisions.append(d)

        print(f"\n{'═'*60}")
        print(f"  ALL DECISIONS SUMMARY")
        print(f"{'═'*60}")
        print(f"  {'Month':<8} {'Retrain':<10} {'Deploy':<10} {'Outcome':<15} Active Model")
        print(f"  {'─'*55}")
        for d in all_decisions:
            if d:
                active = d.get("new_version") if d.get("outcome") == "DEPLOYED" else d.get("active_model")
                print(f"  {d['month']:<8} "
                      f"{'YES' if d['retrain_decision'] else 'no':<10} "
                      f"{str(d.get('deploy_decision','—')):<10} "
                      f"{d.get('outcome','—'):<15} {active}")

    elif args.month:
        engine.run_for_month(args.month)
    else:
        print("Usage:")
        print("  python scripts/decision_engine.py --month 3")
        print("  python scripts/decision_engine.py --all")
        print("  python scripts/decision_engine.py --rollback v1.0.0")
