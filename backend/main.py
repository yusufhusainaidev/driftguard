from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="DriftGuard API",
    description="Adaptive Drift Detection & Retraining for Financial ML Systems",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "DriftGuard API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Routers will be imported here as modules are built
# from routers import drift, models, retraining, predictions
