"""
DriftGuard — Step 2: XGBoost Baseline Model Training
Uses real loan_approval_dataset (Month 1 & 2 batches)

Usage:
    python scripts/train_baseline.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import shap
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_VERSION = "v1.0.0"

CATEGORICAL_COLS = ["education", "self_employed"]
TARGET           = "default_label"
DROP_COLS        = ["month", "drift_injected"]

XGBOOST_PARAMS = {
    "n_estimators":      300,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "eval_metric":       "logloss",
    "random_state":      42,
    "use_label_encoder": False
}


# ─────────────────────────────────────────
# 1. Load Baseline (Month 1 + 2)
# ─────────────────────────────────────────
def load_baseline() -> pd.DataFrame:
    dfs = []
    for m in [1, 2]:
        path = os.path.join(DATA_DIR, f"month_{m}.csv")
        df = pd.read_csv(path)
        dfs.append(df)
        print(f"  Loaded month_{m}.csv: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows | Rejection rate: {combined[TARGET].mean():.2%}")
    return combined


# ─────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    df = df.drop(columns=DROP_COLS, errors="ignore").copy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    return X, y, encoders, feature_cols


# ─────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────
def train(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    return model


# ─────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba), 4),
    }

    print("\n  ── Model Performance ────────────────────")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Approved','Rejected'])}")
    return metrics


# ─────────────────────────────────────────
# 5. SHAP
# ─────────────────────────────────────────
def compute_shap(model, X_train, feature_cols) -> dict:
    print("  Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    mean_abs  = np.abs(shap_values).mean(axis=0)
    shap_dict = dict(zip(feature_cols, mean_abs.tolist()))
    ranked    = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n  ── SHAP Feature Importance ──────────────")
    max_val = max(v for _, v in ranked)
    for rank, (feat, val) in enumerate(ranked, 1):
        bar = "█" * int(val * 40 / max_val)
        print(f"  {rank:2}. {feat:30s} {bar} {val:.4f}")

    return {feat: val for feat, val in ranked}


# ─────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────
def save(model, encoders, shap_importance, metrics, feature_cols):
    model_path = os.path.join(MODEL_DIR, f"model_{MODEL_VERSION}.pkl")
    enc_path   = os.path.join(MODEL_DIR, "encoders.pkl")
    meta_path  = os.path.join(MODEL_DIR, f"metadata_{MODEL_VERSION}.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)

    metadata = {
        "version":         MODEL_VERSION,
        "features":        feature_cols,
        "target":          TARGET,
        "categorical":     CATEGORICAL_COLS,
        "metrics":         metrics,
        "shap_importance": shap_importance,
        "xgboost_params":  {k: v for k, v in XGBOOST_PARAMS.items()
                            if k != "use_label_encoder"}
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model    → {model_path}")
    print(f"  Encoders → {enc_path}")
    print(f"  Metadata → {meta_path}")
    return model_path


# ─────────────────────────────────────────
# 7. MLflow
# ─────────────────────────────────────────
def log_mlflow(model, metrics, shap_importance, model_path):
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("DriftGuard")
        with mlflow.start_run(run_name=f"baseline_{MODEL_VERSION}"):
            mlflow.log_params({k: v for k, v in XGBOOST_PARAMS.items()
                               if k != "use_label_encoder"})
            mlflow.log_param("model_version", MODEL_VERSION)
            mlflow.log_param("training_months", "1,2")
            mlflow.log_param("dataset", "loan_approval_dataset")
            mlflow.log_metrics(metrics)
            for feat, val in shap_importance.items():
                mlflow.log_metric(f"shap_{feat}", round(val, 6))
            mlflow.xgboost.log_model(model, "model")
            mlflow.log_artifact(model_path)
        print(f"  MLflow logged → {MLFLOW_URI}")
    except Exception as e:
        print(f"  ⚠️  MLflow skipped (start Docker first): {e}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DriftGuard — Baseline XGBoost Training")
    print(f"  Version : {MODEL_VERSION}")
    print(f"  Data    : loan_approval_dataset (Month 1+2)")
    print("=" * 55)

    print("\n[1/6] Loading baseline data...")
    df = load_baseline()

    print("\n[2/6] Preprocessing...")
    X, y, encoders, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"  Features: {feature_cols}")

    print("\n[3/6] Training XGBoost...")
    model = train(X_train, y_train, X_test, y_test)

    print("\n[4/6] Evaluating...")
    metrics = evaluate(model, X_test, y_test)

    print("\n[5/6] SHAP analysis...")
    shap_importance = compute_shap(model, X_train, feature_cols)

    print("\n[6/6] Saving...")
    model_path = save(model, encoders, shap_importance, metrics, feature_cols)
    log_mlflow(model, metrics, shap_importance, model_path)

    print("\n" + "=" * 55)
    print("  ✅ Baseline training COMPLETE!")
    print(f"  Accuracy : {metrics['accuracy']}")
    print(f"  F1 Score : {metrics['f1_score']}")
    print(f"  AUC-ROC  : {metrics['auc_roc']}")
    print("\n  Next → Run: python scripts/drift_detection.py")
    print("=" * 55)
