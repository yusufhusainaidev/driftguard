import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DriftGuard Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DriftGuard — Drift Detection Dashboard")
st.caption("Adaptive Drift Detection & Retraining for Financial ML Systems")

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
st.sidebar.header("Controls")
st.sidebar.info("System running. All modules online.")

# ─────────────────────────────────────────
# Placeholder sections — populated as modules are built
# ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric("Active Model", "v1.0.0", "baseline")
col2.metric("Current Accuracy", "—", "")
col3.metric("Drift Status", "Monitoring", "")
col4.metric("Last Retrain", "Never", "")

st.divider()

st.subheader("📊 Drift Score Over Time")
st.info("Drift scores will appear here once batch processing begins.")

st.subheader("📋 Retraining Log")
st.info("Retraining events will appear here.")

st.subheader("🔍 SHAP Feature Importance")
st.info("SHAP explanations will appear here after drift is detected.")
