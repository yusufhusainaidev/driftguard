DriftGuard : Adaptive Drift Detection & Retraining for Financial ML Systems**

DriftGuard is an intelligent system that automatically detects when ML models fail in production, understands why, retrains them safely, and redeploys them without human intervention—specifically designed for financial systems where data changes constantly.

---

## Quick Start

```bash
# 1. Clone & enter project
cd driftguard

# 2. Copy environment file
cp .env.example .env

# 3. Start all services
docker-compose up --build

# 4. Generate synthetic dataset
python scripts/generate_data.py
```

---

## Services

| Service | URL | Description |
|---|---|---|
| FastAPI Backend | http://localhost:8000 | Core system API |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Streamlit Dashboard | http://localhost:8501 | Monitoring UI |
| MLflow | http://localhost:5000 | Model tracking |
| PostgreSQL | localhost:5432 | Main database |
| Redis | localhost:6379 | Distribution cache |

---

## Project Structure

```
driftguard/
├── docker-compose.yml          ← All services
├── .env.example                ← Environment config template
├── backend/
│   ├── Dockerfile
│   ├── main.py                 ← FastAPI app
│   ├── database.py             ← PostgreSQL connection
│   ├── redis_cache.py          ← Redis cache manager
│   └── requirements.txt
├── dashboard/
│   ├── Dockerfile
│   ├── app.py                  ← Streamlit dashboard
│   └── requirements.txt
├── postgres/
│   └── init/
│       └── 01_schema.sql       ← Auto-runs on first start
├── scripts/
│   └── generate_data.py        ← Synthetic dataset generator
└── data/                       ← Generated CSV files go here
```

---

## Database Tables

| Table | Purpose |
|---|---|
| `models` | Model versions, accuracy, status |
| `drift_events` | Detected drift events |
| `retraining_log` | Retrain history & decisions |
| `predictions` | Predictions for A/B testing |
| `shap_explanations` | Feature importance per drift |
| `system_logs` | General activity log |
| `reference_stats` | Baseline stats (Redis backup) |

---

## Team

| Role | Responsibility |
|---|---|
| ML Engineer | XGBoost model, SHAP analysis |
| Drift Engineer | PSI, KS, ADWIN detection |
| Backend | FastAPI endpoints, pipeline |
| Frontend/DevOps | Dashboard, Docker |
