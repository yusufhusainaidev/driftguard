# DriftGuard : **Adaptive Drift Detection & Retraining for Financial ML Systems**
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

## Services

| Service | URL | Description |
|---|---|---|
| FastAPI Backend | http://localhost:8000 | Core system API |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Streamlit Dashboard | http://localhost:8501 | Monitoring UI |
| MLflow | http://localhost:5000 | Model tracking |
| PostgreSQL | localhost:5432 | Main database |
| Redis | localhost:6379 | Distribution cache |

## Project Structure

```
driftguard/
├── .streamlit/
│   └── config.toml
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── redis_cache.py
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── postgres/
│   └── init/
│       └── 01_schema.sql
└── scripts/
    ├── prepare_real_data.py
    ├── train_baseline.py
    ├── run_drift_detection.py
    ├── decision_engine.py
    ├── retraining_pipeline.py
    ├── ab_testing.py
    └── run_demo.py
```

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