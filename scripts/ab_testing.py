# DriftGuard : 6 - A/B Testing
# Compares old model vs new model on a held-out test batch.
# Determines whether to deploy or reject the new model.

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

TARGET    = "default_label"
DROP_COLS = ["date", "month", "drift_type", "loan_id", "loan_status"]
IMPROVEMENT_MARGIN = 0.01


class ABTester:

    def __init__(self):
        self._load_encoders()

    def _load_encoders(self):
        with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
            self.encoders = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "metadata_v1.0.0.json")) as f:
            self.meta = json.load(f)
        self.features = self.meta["features"]

    def _preprocess(self, df: pd.DataFrame) -> tuple:
        df = df.copy()
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col].astype(str))
        X = df[[f for f in self.features if f in df.columns]]
        y = df[TARGET] if TARGET in df.columns else None
        return X, y

    def _load_model(self, version: str):
        path = os.path.join(MODEL_DIR, f"model_{version}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _evaluate(self, model, X, y) -> dict:
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        return {
            "accuracy":  round(float(accuracy_score(y, y_pred)), 4),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y, y_pred, zero_division=0)), 4),
            "f1_score":  round(float(f1_score(y, y_pred, zero_division=0)), 4),
            "auc_roc":   round(float(roc_auc_score(y, y_proba)), 4),
        }

    def run(self, trigger_month: int) -> dict:
        print(f"\n{'='*60}")
        print(f"  A/B Testing — Month {trigger_month}")
        print(f"{'='*60}")

        # Load retrain log
        retrain_log = os.path.join(LOGS_DIR, f"retrain_month_{trigger_month}.json")
        if not os.path.exists(retrain_log):
            print(f"  No retrain log for month {trigger_month}. Run retraining_pipeline first.")
            return {}

        with open(retrain_log) as f:
            retrain_result = json.load(f)

        if retrain_result.get("rl_action") == "none":
            print("  RL chose no action — A/B test skipped.")
            return {"month": trigger_month, "outcome": "SKIPPED", "reason": "RL action was none"}

        # Load test batch — use NEXT month if available, else current
        test_month = min(trigger_month + 1, 6)
        test_path  = os.path.join(DATA_DIR, f"month_{test_month}.csv")
        if not os.path.exists(test_path):
            test_path = os.path.join(DATA_DIR, f"month_{trigger_month}.csv")

        test_df = pd.read_csv(test_path)
        X_test, y_test = self._preprocess(test_df)
        print(f"\n  Test batch: month_{test_month}.csv ({len(test_df)} rows)")

        # Load both models
        control_version   = "v1.0.0"   # original baseline (control group)
        challenger_version = retrain_result.get("new_version")
        if not challenger_version:
            print("  No new version in retrain log.")
            return {}

        control_model   = self._load_model(control_version)
        challenger_model = self._load_model(challenger_version)

        if not control_model or not challenger_model:
            print("  Could not load models for A/B test.")
            return {}

        # Evaluate both
        print(f"\n  Control (A):    {control_version}")
        print(f"  Challenger (B): {challenger_version}")

        control_metrics   = self._evaluate(control_model, X_test, y_test)
        challenger_metrics = self._evaluate(challenger_model, X_test, y_test)

        print(f"\n  ── A/B Results ─────────────────────────────────")
        print(f"  {'Metric':<14} {'Control':<12} {'Challenger':<12} {'Winner'}")
        print(f"  {'─'*50}")
        winners = {}
        for key in ["accuracy","precision","recall","f1_score","auc_roc"]:
            ctrl = control_metrics[key]
            chal = challenger_metrics[key]
            winner = "B ✅" if chal > ctrl else ("A" if ctrl > chal else "TIE")
            winners[key] = winner
            print(f"  {key:<14} {ctrl:<12} {chal:<12} {winner}")

        # Deployment decision
        auc_better = challenger_metrics["auc_roc"] > control_metrics["auc_roc"] + IMPROVEMENT_MARGIN
        f1_better  = challenger_metrics["f1_score"] > control_metrics["f1_score"]
        acc_better = challenger_metrics["accuracy"] > control_metrics["accuracy"]
        challenger_wins = sum([auc_better, f1_better, acc_better])

        if auc_better and (f1_better or acc_better):
            decision = "DEPLOY"
            reason   = (f"Challenger wins on AUC ({control_metrics['auc_roc']}→{challenger_metrics['auc_roc']}) "
                        f"and F1 ({control_metrics['f1_score']}→{challenger_metrics['f1_score']})")
        elif challenger_wins >= 2:
            decision = "DEPLOY"
            reason   = f"Challenger wins on {challenger_wins}/3 primary metrics"
        else:
            decision = "REJECT"
            reason   = (f"Challenger not better enough. "
                        f"AUC: {control_metrics['auc_roc']}→{challenger_metrics['auc_roc']} "
                        f"(required +{IMPROVEMENT_MARGIN})")

        print(f"\n  ── Deployment Decision ─────────────────────────")
        print(f"  Decision : {'✅ DEPLOY' if decision=='DEPLOY' else '❌ REJECT'}")
        print(f"  Reason   : {reason}")

        # Activate if deploying
        if decision == "DEPLOY":
            for f in os.listdir(MODEL_DIR):
                if f.startswith("metadata_") and f.endswith(".json"):
                    path = os.path.join(MODEL_DIR, f)
                    meta = json.load(open(path))
                    if meta["version"] == challenger_version:
                        meta["status"] = "active"
                    elif meta.get("status") == "active":
                        meta["status"] = "retired"
                    with open(path, "w") as fp:
                        json.dump(meta, fp, indent=2)
            print(f"\n  ✅ Model {challenger_version} is now ACTIVE")
        else:
            print(f"\n  ❌ Keeping {control_version} — challenger rejected")

        result = {
            "month":              trigger_month,
            "test_month":         test_month,
            "control_version":    control_version,
            "challenger_version": challenger_version,
            "control_metrics":    control_metrics,
            "challenger_metrics": challenger_metrics,
            "decision":           decision,
            "reason":             reason,
            "timestamp":          datetime.now().isoformat()
        }

        log_path = os.path.join(LOGS_DIR, f"ab_test_month_{trigger_month}.json")
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Log saved → {log_path}")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriftGuard — A/B Testing")
    parser.add_argument("--month", type=int)
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    tester = ABTester()

    if args.all:
        for m in range(1, 7):
            retrain_log = os.path.join(LOGS_DIR, f"retrain_month_{m}.json")
            if os.path.exists(retrain_log):
                tester.run(m)
    elif args.month:
        tester.run(args.month)
    else:
        print("Usage:")
        print("  python scripts/ab_testing.py --month 3")
        print("  python scripts/ab_testing.py --all")