# DriftGuard : 5 - Retraining Pipeline
"""
Integrates Q-Learning RL agent to decide HOW to retrain:
  - none    → keep current model
  - partial → sliding window (recent data only)
  - full    → all available data combined
"""
# The RL agent LEARNS over time which retraining strategy works best for different drift scenarios.


import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


# PATHS & CONFIG

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,  exist_ok=True)

TARGET    = "default_label"
DROP_COLS = ["date", "month", "drift_type", "loan_id", "loan_status"]
WINDOW_SIZE = 1000   # sliding window sample count

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

Q_TABLE_PATH = os.path.join(MODEL_DIR, "q_table.json")



# Q-LEARNING AGENT

class QLearningAgent:
    """
    RL Agent that decides HOW to retrain.

    Actions:
        none    → skip retraining (model is fine)
        partial → sliding window (train on recent N samples)
        full    → full retrain on all available data

    State:
        (drift_type, accuracy_trend)
        drift_type:      "none" | "data_drift" | "concept_drift"
        accuracy_trend:  "improving" | "stable" | "decreasing"

    Reward:
        accuracy improvement after taking the action

    This is what makes DriftGuard novel — the system LEARNS
    which retraining strategy works best over time.
    """

    ACTIONS  = ["none", "partial", "full"]
    ALPHA    = 0.1    # learning rate
    GAMMA    = 0.9    # discount factor
    EPSILON  = 0.2    # exploration rate (20% random, 80% greedy)

    def __init__(self):
        self.Q = self._load_q_table()

    def _load_q_table(self) -> dict:
        if os.path.exists(Q_TABLE_PATH):
            with open(Q_TABLE_PATH) as f:
                return json.load(f)
        return {}

    def save_q_table(self) -> None:
        with open(Q_TABLE_PATH, "w") as f:
            json.dump(self.Q, f, indent=2)

    def get_state(self, drift_type: str, acc_diff: float) -> str:
        """Convert drift + accuracy trend into a state string."""
        if acc_diff > 0.01:
            trend = "improving"
        elif acc_diff < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"
        return f"{drift_type}|{trend}"

    def choose_action(self, state: str) -> str:
        """Epsilon-greedy action selection."""
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.ACTIONS}

        # Exploration
        if random.uniform(0, 1) < self.EPSILON:
            action = random.choice(self.ACTIONS)
            print(f"  [Q-Agent] Exploring → action: {action}")
            return action

        # Exploitation — pick best known action
        action = max(self.Q[state], key=self.Q[state].get)
        print(f"  [Q-Agent] Exploiting → best action: {action} "
              f"(Q={self.Q[state][action]:.4f})")
        return action

    def update(self, state: str, action: str,
               reward: float, next_state: str) -> None:
        """Update Q-table using Bellman equation."""
        if next_state not in self.Q:
            self.Q[next_state] = {a: 0.0 for a in self.ACTIONS}

        old_val    = self.Q[state][action]
        future_max = max(self.Q[next_state].values())
        new_val    = old_val + self.ALPHA * (reward + self.GAMMA * future_max - old_val)

        self.Q[state][action] = new_val
        self.save_q_table()

        print(f"  [Q-Agent] Q[{state}][{action}]: {old_val:.4f} → {new_val:.4f} "
              f"(reward={reward:+.4f})")

    def print_q_table(self) -> None:
        print("\n  ── Q-Table ────────────────────────────────")
        for state, actions in self.Q.items():
            best = max(actions, key=actions.get)
            print(f"  {state}")
            for a, v in actions.items():
                marker = " ← best" if a == best else ""
                print(f"    {a:10s}: {v:+.4f}{marker}")



# Data Preprocessor

class DataPreprocessor:
    def __init__(self):
        enc_path = os.path.join(MODEL_DIR, "encoders.pkl")
        with open(enc_path, "rb") as f:
            self.encoders = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "metadata_v1.0.0.json")) as f:
            self.meta = json.load(f)
        self.features = self.meta["features"]

    def process(self, df: pd.DataFrame) -> tuple:
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



# Retraining Strategies

class RetrainingStrategies:

    @staticmethod
    def sliding_window(X_all: np.ndarray, y_all: np.ndarray) -> tuple:
        """Use only the most recent N samples."""
        if len(X_all) > WINDOW_SIZE:
            X = X_all[-WINDOW_SIZE:]
            y = y_all[-WINDOW_SIZE:]
            print(f"  Sliding window: using last {WINDOW_SIZE} of {len(X_all)} samples")
        else:
            X, y = X_all, y_all
        return X, y

    @staticmethod
    def full_retrain(X_all: np.ndarray, y_all: np.ndarray) -> tuple:
        """Use all available data."""
        print(f"  Full retrain: using all {len(X_all)} samples")
        return X_all, y_all

    @staticmethod
    def train(X, y) -> tuple:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc_roc":  round(float(roc_auc_score(y_test, y_proba)), 4),
        }
        return model, metrics



# Main Retraining Pipeline

class RetrainingPipeline:

    def __init__(self):
        self.agent      = QLearningAgent()
        self.preprocessor = DataPreprocessor()
        self.strategies = RetrainingStrategies()
        self._load_active_model()

    def _load_active_model(self):
        # Pass 1: find explicitly active model
        for f in sorted(os.listdir(MODEL_DIR), reverse=True):
            if f.startswith("metadata_") and f.endswith(".json"):
                meta = json.load(open(os.path.join(MODEL_DIR, f)))
                if meta.get("status") == "active":
                    self._apply_model(meta)
                    return

        # Pass 2: fallback — force v1.0.0 active
        v1_path = os.path.join(MODEL_DIR, "metadata_v1.0.0.json")
        if os.path.exists(v1_path):
            meta = json.load(open(v1_path))
            meta["status"] = "active"
            with open(v1_path, "w") as fp:
                json.dump(meta, fp, indent=2)
            print("  [Auto-fixed] v1.0.0 set as active.")
            self._apply_model(meta)
            return

        raise RuntimeError("No active model found. Run train_baseline.py first.")

    def _apply_model(self, meta):
        self.active_meta    = meta
        self.active_version = meta["version"]
        self.active_metrics = meta["metrics"]
        model_path = os.path.join(MODEL_DIR, f"model_{meta['version']}.pkl")
        with open(model_path, "rb") as fp:
            self.active_model = pickle.load(fp)
        print(f"  Active model: {self.active_version} | Accuracy: {self.active_metrics['accuracy']}")

    def _load_all_data_up_to(self, month: int) -> tuple:
        """Load all available data up to given month."""
        dfs = []
        for m in range(1, month + 1):
            path = os.path.join(DATA_DIR, f"month_{m}.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        combined = pd.concat(dfs, ignore_index=True)
        return self.preprocessor.process(combined)

    def _get_drift_type(self, month: int) -> str:
        drift_log = os.path.join(LOGS_DIR, f"drift_month_{month}.json")
        if not os.path.exists(drift_log):
            return "none"
        with open(drift_log) as f:
            d = json.load(f)
        if d.get("concept_drift_detected"):
            return "concept_drift"
        if d.get("data_drift_detected"):
            return "data_drift"
        return "none"

    def _save_new_model(self, model, metrics, version, trigger_month, action, old_metrics):
        model_path = os.path.join(MODEL_DIR, f"model_{version}.pkl")
        meta_path  = os.path.join(MODEL_DIR, f"metadata_{version}.json")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        metadata = {
            "version":         version,
            "features":        self.preprocessor.features,
            "target":          TARGET,
            "metrics":         metrics,
            "old_metrics":     old_metrics,
            "trigger_month":   trigger_month,
            "rl_action":       action,
            "xgboost_params":  XGBOOST_PARAMS,
            "status":          "candidate",
            "trained_at":      datetime.now().isoformat()
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return model_path

    def _activate_model(self, version: str):
        for f in os.listdir(MODEL_DIR):
            if f.startswith("metadata_") and f.endswith(".json"):
                path = os.path.join(MODEL_DIR, f)
                meta = json.load(open(path))
                meta["status"] = "active" if meta["version"] == version else (
                    "retired" if meta.get("status") == "active" else meta.get("status", "retired")
                )
                with open(path, "w") as fp:
                    json.dump(meta, fp, indent=2)

    def run(self, trigger_month: int) -> dict:
        print(f"\n{'='*60}")
        print(f"  Retraining Pipeline — Month {trigger_month}")
        print(f"{'='*60}")

        # Check if retrain was triggered
        dec_log = os.path.join(LOGS_DIR, f"decision_month_{trigger_month}.json")
        if os.path.exists(dec_log):
            dec = json.load(open(dec_log))
            if not dec.get("retrain_decision"):
                print(f"  No retrain needed for Month {trigger_month}.")
                return {"month": trigger_month, "action": "none", "outcome": "SKIPPED"}

        drift_type = self._get_drift_type(trigger_month)
        prev_acc   = self.active_metrics.get("accuracy", 0.97)

        # Step 1: RL state
        acc_diff = 0
        state    = self.agent.get_state(drift_type, acc_diff)
        print(f"\n[RL] Drift type: {drift_type}")
        print(f"[RL] Current state: {state}")

        # Step 2: Choose action
        action = self.agent.choose_action(state)
        print(f"[RL] Selected action: {action.upper()}")

        # Step 3: Execute action
        X_all, y_all = self._load_all_data_up_to(trigger_month)
        X_np = X_all.values if hasattr(X_all, "values") else X_all
        y_np = y_all.values if hasattr(y_all, "values") else y_all

        if action == "none":
            print("\n  Action: NONE — keeping current model")
            new_acc     = prev_acc
            new_metrics = self.active_metrics
            new_model   = self.active_model
            deployed    = False
            new_version = self.active_version

        elif action == "partial":
            print("\n  Action: PARTIAL — sliding window retrain")
            X_sw, y_sw = self.strategies.sliding_window(X_np, y_np)
            new_model, new_metrics = self.strategies.train(X_sw, y_sw)
            new_acc     = new_metrics["accuracy"]
            major       = int(self.active_version.lstrip("v").split(".")[0])
            new_version = f"v{major + 1}.0.0"
            self._save_new_model(new_model, new_metrics, new_version, trigger_month, action, self.active_metrics)
            deployed = new_acc > prev_acc
            if deployed:
                self._activate_model(new_version)
                self.active_version = new_version
                self.active_metrics = new_metrics

        elif action == "full":
            print("\n  Action: FULL — complete retrain on all data")
            X_f, y_f = self.strategies.full_retrain(X_np, y_np)
            new_model, new_metrics = self.strategies.train(X_f, y_f)
            new_acc     = new_metrics["accuracy"]
            major       = int(self.active_version.lstrip("v").split(".")[0])
            new_version = f"v{major + 1}.0.0"
            self._save_new_model(new_model, new_metrics, new_version, trigger_month, action, self.active_metrics)
            deployed = new_acc > prev_acc
            if deployed:
                self._activate_model(new_version)
                self.active_version = new_version
                self.active_metrics = new_metrics

        # Step 4: Compute reward
        reward     = new_acc - prev_acc
        acc_diff_r = reward
        next_state = self.agent.get_state(drift_type, acc_diff_r)

        # Step 5: Update Q-table
        print(f"\n[RL] Reward: {reward:+.4f}")
        self.agent.update(state, action, reward, next_state)

        # Summary
        print(f"\n{'─'*60}")
        print(f"  RETRAINING SUMMARY")
        print(f"{'─'*60}")
        print(f"  Action taken    : {action.upper()}")
        print(f"  Previous Acc    : {prev_acc:.4f}")
        print(f"  New Acc         : {new_acc:.4f}")
        print(f"  Improvement     : {reward:+.4f}")
        print(f"  Model deployed  : {'✅ YES' if deployed else '❌ NO'}")
        print(f"  Active model    : {self.active_version}")

        if len(self.agent.Q) >= 2:
            self.agent.print_q_table()

        result = {
            "month":        trigger_month,
            "drift_type":   drift_type,
            "rl_state":     state,
            "rl_action":    action,
            "prev_accuracy":prev_acc,
            "new_accuracy": new_acc,
            "reward":       round(reward, 4),
            "deployed":     deployed,
            "new_version":  new_version,
            "new_metrics":  new_metrics,
            "timestamp":    datetime.now().isoformat()
        }

        log_path = os.path.join(LOGS_DIR, f"retrain_month_{trigger_month}.json")
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Log saved → {log_path}")

        return result



# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DriftGuard — Retraining Pipeline")
    parser.add_argument("--month", type=int, help="Trigger month")
    parser.add_argument("--all",   action="store_true", help="Run all months")
    args = parser.parse_args()

    pipeline = RetrainingPipeline()

    if args.all:
        results = []
        for m in range(1, 7):
            dec_log = os.path.join(LOGS_DIR, f"decision_month_{m}.json")
            if not os.path.exists(dec_log):
                print(f"  Skipping month {m} — run drift detection first")
                continue
            dec = json.load(open(dec_log))
            if dec.get("retrain_decision"):
                r = pipeline.run(m)
                results.append(r)

        print(f"\n{'═'*60}")
        print(f"  Q-LEARNING SUMMARY — All Triggered Retrains")
        print(f"{'═'*60}")
        print(f"  {'Month':<8} {'Action':<12} {'Old Acc':<12} {'New Acc':<12} {'Reward':<10} Deployed")
        print(f"  {'─'*55}")
        for r in results:
            print(f"  {r['month']:<8} {r['rl_action']:<12} {r['prev_accuracy']:<12} "
                  f"{r['new_accuracy']:<12} {r['reward']:<10.4f} {'✅' if r['deployed'] else '❌'}")

    elif args.month:
        pipeline.run(args.month)
    else:
        print("Usage:")
        print("  python scripts/retraining_pipeline.py --month 3")
        print("  python scripts/retraining_pipeline.py --all")