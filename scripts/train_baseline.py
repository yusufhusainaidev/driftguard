"""
DriftGuard — Step 2: XGBoost Baseline Model Training
Trains on Month 1 & 2 (real loan approval data, clean baseline)
Saves model locally + logs to MLflow

Usage:
    python scripts/train_baseline.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing     import LabelEncoder

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
BASELINE_FILES = ["month_1.csv", "month_2.csv"]
MODEL_VERSION  = "v1.0.0"
TARGET         = "default_label"

XGBOOST_PARAMS = {
    "n_estimators":      300,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "scale_pos_weight":  1,
    
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1
}

DROP_COLS = ["loan_id", "loan_status", "date", "month", "drift_type", TARGET]


# ─────────────────────────────────────────
# 1. Load Baseline Data
# ─────────────────────────────────────────
def load_baseline() -> pd.DataFrame:
    dfs = []
    for fname in BASELINE_FILES:
        path = os.path.join(DATA_DIR, fname)
        df   = pd.read_csv(path)
        dfs.append(df)
        print(f"  {fname}: {len(df)} rows | default: {df[TARGET].mean():.2%}")
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────
def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()

    drop = [c for c in DROP_COLS if c in df.columns and c != TARGET]
    df   = df.drop(columns=drop)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    cat_cols = ["education", "self_employed"]
    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET]

    return X, y, encoders, features


# ─────────────────────────────────────────
# 3. Train XGBoost
# ─────────────────────────────────────────
def train(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


# ─────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────
def evaluate(model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc":   round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    print("\n  ── Model Performance ──────────────────────────")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v}")

    print(f"\n{classification_report(y_test, y_pred, target_names=['Approved','Rejected'])}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}")

    return metrics


# ─────────────────────────────────────────
# 5. Cross-Validation
# ─────────────────────────────────────────
def cross_validate(X, y) -> float:
    model  = xgb.XGBClassifier(**XGBOOST_PARAMS)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n  Cross-Validation AUC-ROC (5-fold):")
    print(f"    Scores : {[round(s, 4) for s in scores]}")
    print(f"    Mean   : {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean())


# ─────────────────────────────────────────
# 6. SHAP Feature Importance
# ─────────────────────────────────────────
def compute_shap(model, X_train, features) -> dict:
    print("\n  Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    mean_abs    = np.abs(shap_values).mean(axis=0)
    shap_dict   = dict(zip(features, mean_abs.tolist()))
    ranked      = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)

    print("\n  ── SHAP Feature Importance ────────────────────")
    max_val = max(v for _, v in ranked)
    for rank, (feat, val) in enumerate(ranked, 1):
        bar = "█" * int(val * 30 / max_val)
        print(f"    {rank}. {feat:28s}: {val:.4f}  {bar}")

    return {feat: val for feat, val in ranked}


# ─────────────────────────────────────────
# 7. Save Locally
# ─────────────────────────────────────────
def save_locally(model, encoders, shap_importance, metrics, features, cv_score):
    model_path = os.path.join(MODEL_DIR, f"model_{MODEL_VERSION}.pkl")
    enc_path   = os.path.join(MODEL_DIR, "encoders.pkl")
    meta_path  = os.path.join(MODEL_DIR, f"metadata_{MODEL_VERSION}.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)

    metadata = {
        "version":         MODEL_VERSION,
        "features":        features,
        "target":          TARGET,
        "metrics":         metrics,
        "cv_auc_roc":      cv_score,
        "shap_importance": shap_importance,
        "baseline_files":  BASELINE_FILES,
        "xgboost_params":  XGBOOST_PARAMS,
        "status":          "active"
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Model    → {model_path}")
    print(f"  Encoders → {enc_path}")
    print(f"  Metadata → {meta_path}")
    return model_path


# ─────────────────────────────────────────
# 8. MLflow Logging
# ─────────────────────────────────────────
def log_mlflow(model, metrics, shap_importance, cv_score, model_path):
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("DriftGuard")
        with mlflow.start_run(run_name=f"baseline_{MODEL_VERSION}"):
            mlflow.log_params({**XGBOOST_PARAMS, "model_version": MODEL_VERSION})
            mlflow.log_metrics({**metrics, "cv_auc_roc": cv_score})
            for feat, val in shap_importance.items():
                mlflow.log_metric(f"shap_{feat}", round(val, 6))
            mlflow.xgboost.log_model(model, "model")
            mlflow.log_artifact(model_path)
        print(f"\n  MLflow run logged at: {MLFLOW_URI}")
    except Exception as e:
        print(f"\n  MLflow offline — model saved locally only.")
        print(f"  (Run docker-compose up to enable MLflow tracking)")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  DriftGuard — Baseline Model Training")
    print(f"  Version : {MODEL_VERSION}")
    print(f"  Source  : loan_approval_dataset.csv (Month 1 & 2)")
    print("=" * 60)

    print("\n[1/7] Loading baseline data (Month 1 & 2)...")
    df = load_baseline()
    print(f"  Total: {len(df)} rows | overall default: {df[TARGET].mean():.2%}")

    print("\n[2/7] Preprocessing...")
    X, y, encoders, features = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Features ({len(features)}): {features}")

    print("\n[3/7] Cross-validation (5-fold)...")
    cv_score = cross_validate(X, y)

    print("\n[4/7] Training XGBoost...")
    model = train(X_train, y_train, X_test, y_test)
    print("  Done.")

    print("\n[5/7] Evaluating...")
    metrics = evaluate(model, X_test, y_test)

    print("\n[6/7] Computing SHAP...")
    shap_importance = compute_shap(model, X_train, features)

    print("\n[7/7] Saving...")
    model_path = save_locally(model, encoders, shap_importance, metrics, features, cv_score)
    log_mlflow(model, metrics, shap_importance, cv_score, model_path)

    print("\n" + "=" * 60)
    print("  ✅ Baseline model training COMPLETE!")
    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  AUC-ROC   : {metrics['auc_roc']}")
    print(f"  CV AUC    : {cv_score:.4f}")
    print(f"  Top SHAP  : {list(shap_importance.keys())[0]}")
    print("\n  Next → python scripts/run_drift_detection.py")
    print("=" * 60)
