#DriftGuard
"""
Run:
    uvicorn backend.main:app --reload --port 8000
    OR 
    from project root: python -m uvicorn backend.main:app --reload
"""

import os, sys, json, subprocess
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

app = FastAPI(
    title="DriftGuard API",
    description="Adaptive Drift Detection & Retraining for Financial ML Systems",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None

def _active_meta():
    if not os.path.exists(MODEL_DIR): return None
    for f in sorted(os.listdir(MODEL_DIR), reverse=True):
        if f.startswith("metadata_") and f.endswith(".json"):
            m = _load(os.path.join(MODEL_DIR, f))
            if m and m.get("status") == "active": return m
    return None

def _run(script, args=[]):
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, script)] + args,
                       capture_output=True, text=True, cwd=BASE_DIR)
    return {"success": r.returncode==0, "stdout": r.stdout[-2000:], "stderr": r.stderr[-500:]}


# System
@app.get("/", tags=["System"])
def root():
    return {"system": "DriftGuard", "version": "1.0.0", "status": "running", "docs": "/docs"}

@app.get("/health", tags=["System"])
def health():
    m = _active_meta()
    return {"status": "healthy", "active_model": m["version"] if m else None,
            "accuracy": m["metrics"]["accuracy"] if m else None,
            "timestamp": datetime.now().isoformat()}


# Model
@app.get("/model/active", tags=["Model"])
def active_model():
    m = _active_meta()
    if not m: raise HTTPException(404, "No active model")
    return m

@app.get("/model/all", tags=["Model"])
def all_models():
    if not os.path.exists(MODEL_DIR): return {"models": []}
    models = []
    for f in sorted(os.listdir(MODEL_DIR)):
        if f.startswith("metadata_") and f.endswith(".json"):
            m = _load(os.path.join(MODEL_DIR, f))
            if m: models.append({"version": m["version"], "status": m["status"],
                                  "accuracy": m.get("metrics",{}).get("accuracy"),
                                  "auc_roc":  m.get("metrics",{}).get("auc_roc")})
    return {"models": models, "count": len(models)}

@app.get("/model/shap", tags=["Model"])
def shap():
    m = _active_meta()
    if not m or "shap_importance" not in m: raise HTTPException(404, "SHAP not available")
    ranked = sorted(m["shap_importance"].items(), key=lambda x: x[1], reverse=True)
    return {"model_version": m["version"],
            "shap": [{"rank": i+1, "feature": f, "value": round(v,6)} for i,(f,v) in enumerate(ranked)]}

@app.post("/model/rollback/{version}", tags=["Model"])
def rollback(version: str):
    if not os.path.exists(os.path.join(MODEL_DIR, f"model_{version}.pkl")):
        raise HTTPException(404, f"Model {version} not found")
    for f in os.listdir(MODEL_DIR):
        if f.startswith("metadata_") and f.endswith(".json"):
            path = os.path.join(MODEL_DIR, f)
            m = json.load(open(path))
            m["status"] = "active" if m["version"]==version else ("retired" if m.get("status")=="active" else m.get("status","retired"))
            json.dump(m, open(path,"w"), indent=2)
    return {"message": f"Rolled back to {version}", "timestamp": datetime.now().isoformat()}


# Drift
@app.get("/drift/summary", tags=["Drift"])
def drift_summary():
    d = _load(os.path.join(LOGS_DIR, "drift_summary.json"))
    if not d: raise HTTPException(404, "No drift summary. Run detection first.")
    return {"summary": d}

@app.get("/drift/{month}", tags=["Drift"])
def drift_month(month: int):
    if not 1<=month<=6: raise HTTPException(400, "Month 1-6 only")
    d = _load(os.path.join(LOGS_DIR, f"drift_month_{month}.json"))
    if not d: raise HTTPException(404, f"No drift data for month {month}")
    return d

@app.post("/drift/detect/{month}", tags=["Drift"])
def detect_drift(month: int, bg: BackgroundTasks):
    if not 1<=month<=6: raise HTTPException(400, "Month 1-6 only")
    bg.add_task(_run, "run_drift_detection.py", ["--month", str(month)])
    return {"message": f"Drift detection started for month {month}", "status": "running"}


# Decisions
@app.get("/decisions", tags=["Decisions"])
def decisions():
    return {"decisions": [d for m in range(1,7) for d in [_load(os.path.join(LOGS_DIR, f"decision_month_{m}.json"))] if d]}

@app.get("/decisions/{month}", tags=["Decisions"])
def decision_month(month: int):
    d = _load(os.path.join(LOGS_DIR, f"decision_month_{month}.json"))
    if not d: raise HTTPException(404, f"No decision for month {month}")
    return d


# Retraining
@app.post("/retrain/{month}", tags=["Retraining"])
def retrain(month: int):
    if not 1<=month<=6: raise HTTPException(400, "Month 1-6 only")
    r = _run("retraining_pipeline.py", ["--month", str(month)])
    if not r["success"]: raise HTTPException(500, f"Retrain failed: {r['stderr']}")
    return {"message": f"Retrain complete", "result": _load(os.path.join(LOGS_DIR, f"retrain_month_{month}.json"))}

@app.get("/retrain/{month}", tags=["Retraining"])
def retrain_log(month: int):
    d = _load(os.path.join(LOGS_DIR, f"retrain_month_{month}.json"))
    if not d: raise HTTPException(404, f"No retrain log for month {month}")
    return d


# A/B Testing
@app.post("/ab_test/run/{month}", tags=["A/B Test"])
def run_ab(month: int):
    r = _run("ab_testing.py", ["--month", str(month)])
    if not r["success"]: raise HTTPException(500, f"A/B failed: {r['stderr']}")
    return {"result": _load(os.path.join(LOGS_DIR, f"ab_test_month_{month}.json"))}

@app.get("/ab_test/{month}", tags=["A/B Test"])
def get_ab(month: int):
    d = _load(os.path.join(LOGS_DIR, f"ab_test_month_{month}.json"))
    if not d: raise HTTPException(404, f"No A/B test for month {month}")
    return d


# Full Pipeline
@app.post("/pipeline/run/{month}", tags=["Pipeline"])
def pipeline_month(month: int):
    steps = {}
    steps["drift"]    = _run("run_drift_detection.py",  ["--month", str(month)])
    steps["decision"] = _run("decision_engine.py",      ["--month", str(month)])
    dec = _load(os.path.join(LOGS_DIR, f"decision_month_{month}.json"))
    if dec and dec.get("retrain_decision"):
        steps["retrain"]  = _run("retraining_pipeline.py", ["--month", str(month)])
        steps["ab_test"]  = _run("ab_testing.py",           ["--month", str(month)])
    m = _active_meta()
    return {
        "month":        month,
        "steps":        {k: v["success"] for k,v in steps.items()},
        "drift":        _load(os.path.join(LOGS_DIR, f"drift_month_{month}.json")),
        "active_model": m["version"] if m else None,
        "accuracy":     m["metrics"]["accuracy"] if m else None,
    }

@app.post("/pipeline/run_all", tags=["Pipeline"])
def pipeline_all(bg: BackgroundTasks):
    def _all():
        for m in range(1,7):
            _run("run_drift_detection.py", ["--month",str(m)])
            _run("decision_engine.py",     ["--month",str(m)])
            dec = _load(os.path.join(LOGS_DIR,f"decision_month_{m}.json"))
            if dec and dec.get("retrain_decision"):
                _run("retraining_pipeline.py",["--month",str(m)])
                _run("ab_testing.py",         ["--month",str(m)])
    bg.add_task(_all)
    return {"message": "Full pipeline running in background for all months"}