"""
DriftGuard — Dashboard

Run:
    streamlit run dashboard/app.py
"""

import os, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# PATHS
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR  = os.path.join(BASE_DIR, "logs")

# PAGE CONFIG
st.set_page_config(
    page_title="DriftGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0a0e17;--card:#0f1623;--raised:#161e2e;--border:#1e2d45;
      --green:#00ff88;--red:#ff3b5c;--yellow:#ffbe00;--blue:#3b82f6;--muted:#64748b;--text:#e8edf5;}
.stApp{background:var(--bg);}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text);}
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem;}

/* Tab styling */
.stTabs [data-baseweb="tab-list"]{
    gap:4px; background:var(--card); padding:6px;
    border-radius:8px; border:1px solid var(--border);
    margin-bottom:20px;
}
.stTabs [data-baseweb="tab"]{
    background:transparent; color:var(--muted);
    border-radius:6px; padding:8px 18px;
    font-family:'DM Sans',sans-serif; font-size:.85rem;
    border:none; transition:all .2s;
}
.stTabs [aria-selected="true"]{
    background:var(--raised)!important; color:var(--text)!important;
    border:1px solid var(--border)!important;
}
.stTabs [data-baseweb="tab-panel"]{padding-top:0;}

/* Cards */
.dg-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px 20px;margin-bottom:12px;}
.card-g{border-left:3px solid var(--green);} .card-r{border-left:3px solid var(--red);}
.card-y{border-left:3px solid var(--yellow);} .card-b{border-left:3px solid var(--blue);}

/* Metrics grid */
.mg{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;}
.mc{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:18px;text-align:center;}
.mv{font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;line-height:1;margin-bottom:5px;}
.ml{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;}
.green{color:var(--green);} .red{color:var(--red);} .yellow{color:var(--yellow);} .blue{color:var(--blue);}

/* Section label */
.sec{font-family:'Space Mono',monospace;font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;margin-bottom:9px;}
.hr{border:none;border-top:1px solid var(--border);margin:16px 0;}

/* Badges */
.badge{display:inline-block;padding:2px 9px;border-radius:3px;font-family:'Space Mono',monospace;font-size:.68rem;font-weight:700;}
.bg{background:rgba(0,255,136,.1);color:var(--green);border:1px solid rgba(0,255,136,.3);}
.br{background:rgba(255,59,92,.1);color:var(--red);border:1px solid rgba(255,59,92,.3);}
.by{background:rgba(255,190,0,.1);color:var(--yellow);border:1px solid rgba(255,190,0,.3);}
.bb{background:rgba(59,130,246,.1);color:var(--blue);border:1px solid rgba(59,130,246,.3);}

/* Table */
.dtable{width:100%;border-collapse:collapse;font-size:.82rem;}
.dtable th{font-family:'Space Mono',monospace;font-size:.64rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);padding:9px 11px;border-bottom:1px solid var(--border);text-align:left;}
.dtable td{padding:9px 11px;border-bottom:1px solid rgba(30,45,69,.4);}

/* SHAP bars */
.shap-row{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.shap-lbl{font-size:.8rem;width:160px;flex-shrink:0;}
.shap-bg{flex:1;background:var(--raised);border-radius:3px;height:8px;overflow:hidden;}
.shap-fil{height:100%;border-radius:3px;}
.shap-val{font-family:'Space Mono',monospace;font-size:.72rem;color:var(--muted);width:55px;text-align:right;}

/* RL Q-table cell */
.q-cell{padding:8px 12px;border-radius:4px;text-align:center;font-family:'Space Mono',monospace;font-size:.8rem;}

/* Alert boxes */
.alert-ok {background:rgba(0,255,136,.06);border:1px solid rgba(0,255,136,.2);border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:.83rem;}
.alert-bad{background:rgba(255,59,92,.07);border:1px solid rgba(255,59,92,.25);border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:.83rem;}
</style>
""", unsafe_allow_html=True)

# DATA LOADERS
def _load(path):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except: pass
    return None

@st.cache_data(ttl=10)
def load_drift(m):   return _load(os.path.join(LOGS_DIR, f"drift_month_{m}.json"))
@st.cache_data(ttl=10)
def load_dec(m):     return _load(os.path.join(LOGS_DIR, f"decision_month_{m}.json"))
@st.cache_data(ttl=10)
def load_summary():  return _load(os.path.join(LOGS_DIR, "drift_summary.json")) or []
@st.cache_data(ttl=10)
def load_retrain(m): return _load(os.path.join(LOGS_DIR, f"retrain_month_{m}.json"))
@st.cache_data(ttl=10)
def load_ab(m):      return _load(os.path.join(LOGS_DIR, f"ab_test_month_{m}.json"))

@st.cache_data(ttl=10)
def load_q_table():
    return _load(os.path.join(MODEL_DIR, "q_table.json")) or {}

@st.cache_data(ttl=10)
def load_meta_for_shap():
    """Always load v1.0.0 for SHAP — only baseline has SHAP values."""
    m = _load(os.path.join(MODEL_DIR, "metadata_v1.0.0.json"))
    if m and "shap_importance" in m and m["shap_importance"]:
        return m
    # Fallback: scan all metadata files for one with shap_importance
    if os.path.exists(MODEL_DIR):
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.startswith("metadata_") and f.endswith(".json"):
                meta = _load(os.path.join(MODEL_DIR, f))
                if meta and meta.get("shap_importance"):
                    return meta
    return None

@st.cache_data(ttl=10)
def load_active_meta():
    if not os.path.exists(MODEL_DIR):
        return None
    # Try active first
    for f in sorted(os.listdir(MODEL_DIR), reverse=True):
        if f.startswith("metadata_") and f.endswith(".json"):
            m = _load(os.path.join(MODEL_DIR, f))
            if m and m.get("status") == "active":
                return m
    # Fallback: v1.0.0
    return _load(os.path.join(MODEL_DIR, "metadata_v1.0.0.json"))

@st.cache_data(ttl=10)
def load_df():
    p = os.path.join(DATA_DIR, "full_dataset.csv")
    if os.path.exists(p):
        try: return pd.read_csv(p)
        except: pass
    return None

# PLOTLY LAYOUT
BL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e8edf5", size=12),
    margin=dict(l=12, r=12, t=32, b=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d45")
)
XA = dict(showgrid=False, zeroline=False, color="#64748b", linecolor="#1e2d45", tickcolor="#1e2d45")
YA = dict(showgrid=True, gridcolor="#1e2d45", zeroline=False, color="#64748b", linecolor="#1e2d45")

def pchart(fig, height=300, xa=None, ya=None):
    fig.update_layout(**BL, height=height)
    fig.update_xaxes(**{**XA, **(xa or {})})
    fig.update_yaxes(**{**YA, **(ya or {})})
    st.plotly_chart(fig, use_container_width=True)

# SIDEBAR — system status only
with st.sidebar:
    st.markdown("""<div style='padding:6px 0 16px 0;'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;color:#00ff88;'>🛡️ DriftGuard</div>
        <div style='font-size:.7rem;color:#64748b;margin-top:3px;'>Financial ML Monitor</div>
    </div>""", unsafe_allow_html=True)

    meta = load_active_meta()
    if meta:
        st.markdown(f"""<div style='font-size:.78rem;'>
            <div style='color:#64748b;margin-bottom:3px;font-family:Space Mono,monospace;font-size:.62rem;'>ACTIVE MODEL</div>
            <div style='font-family:Space Mono,monospace;color:#00ff88;font-size:.9rem;margin-bottom:10px;'>{meta.get('version','—')}</div>
            <div style='color:#64748b;margin-bottom:3px;font-family:Space Mono,monospace;font-size:.62rem;'>AUC-ROC</div>
            <div style='font-family:Space Mono,monospace;color:#e8edf5;margin-bottom:10px;'>{meta.get('metrics',{}).get('auc_roc','—')}</div>
            <div style='color:#64748b;margin-bottom:3px;font-family:Space Mono,monospace;font-size:.62rem;'>ACCURACY</div>
            <div style='font-family:Space Mono,monospace;color:#e8edf5;margin-bottom:10px;'>{meta.get('metrics',{}).get('accuracy','—')}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("No model found. Run train_baseline.py")

    st.markdown("<hr style='border-color:#1e2d45;margin:12px 0;'>", unsafe_allow_html=True)

    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"<div style='font-size:.68rem;color:#64748b;margin-top:8px;'>Updated {datetime.now().strftime('%H:%M:%S')}</div>",
                unsafe_allow_html=True)

    st.markdown("""<hr style='border-color:#1e2d45;margin:12px 0;'>
    <div style='font-size:.72rem;color:#64748b;'>
        <div style='margin-bottom:4px;'>📌 <b>Navigation</b></div>
        <div>Use the tabs above the</div>
        <div>main content area to</div>
        <div>switch between pages.</div>
    </div>""", unsafe_allow_html=True)

# MAIN TABS NAVIGATION
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Drift Monitor",
    "SHAP",
    "RL Analysis",
    "Decision Log",
    "Data Explorer"
])


# TAB 1 — OVERVIEW

with tab1:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;margin-bottom:3px;'>System Overview</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.82rem;color:#64748b;margin-bottom:20px;'>DriftGuard · Adaptive Drift Detection & Retraining for Financial ML</div>", unsafe_allow_html=True)

    summary = load_summary()
    meta    = load_active_meta()
    nd = sum(1 for r in summary if r.get("any_drift_detected"))
    nr = sum(1 for r in summary if r.get("trigger_retrain"))

    acc = meta.get('metrics',{}).get('accuracy','—') if meta else '—'
    auc = meta.get('metrics',{}).get('auc_roc','—') if meta else '—'

    st.markdown(f"""<div class='mg'>
        <div class='mc card-g'><div class='mv green'>{acc}</div><div class='ml'>Accuracy</div></div>
        <div class='mc card-b'><div class='mv blue'>{auc}</div><div class='ml'>AUC-ROC</div></div>
        <div class='mc card-{"r" if nd else "g"}'><div class='mv {"red" if nd else "green"}'>{nd}</div><div class='ml'>Drift Events</div></div>
        <div class='mc card-y'><div class='mv yellow'>{nr}</div><div class='ml'>Retrain Triggers</div></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("<div class='sec'>Accuracy Trend — Monthly</div>", unsafe_allow_html=True)
        months_l, accs, flags = [], [], []
        for m in range(1,7):
            d = load_drift(m)
            if d:
                months_l.append(f"M{m}")
                acc_v = d.get("concept_drift",{}).get("accuracy_check",{}).get("current_accuracy")
                accs.append(acc_v if acc_v else (meta.get("metrics",{}).get("accuracy",0.97) if meta else 0.97))
                flags.append(d.get("any_drift_detected",False))
        if months_l:
            fig = go.Figure()
            for i, flag in enumerate(flags):
                if flag: fig.add_vrect(x0=i-.4, x1=i+.4, fillcolor="rgba(255,59,92,.07)", line_width=0, layer="below")
            fig.add_trace(go.Scatter(x=months_l, y=accs, mode="lines+markers",
                line=dict(color="#3b82f6",width=2.5),
                marker=dict(size=9, color=["#ff3b5c" if f else "#00ff88" for f in flags], line=dict(width=2,color="#0f1623")),
                fill="tozeroy", fillcolor="rgba(59,130,246,.04)", name="Accuracy"))
            if meta:
                bline = meta.get("metrics",{}).get("accuracy",0.97)
                fig.add_hline(y=bline, line_dash="dot", line_color="rgba(0,255,136,.35)",
                              annotation_text=f"Baseline {bline}", annotation_font_color="#00ff88", annotation_font_size=10)
            pchart(fig, 250, ya={"range":[.75,1.02]})
        else:
            st.info("Run drift detection to see trend.")

    with c2:
        st.markdown("<div class='sec'>Drift Timeline</div>", unsafe_allow_html=True)
        for m in range(1,7):
            d   = load_drift(m)
            dec = load_dec(m)
            if not d: 
                st.markdown(f"""<div style='padding:7px 11px;background:var(--card);border:1px solid var(--border);border-radius:5px;margin-bottom:5px;font-size:.75rem;color:#64748b;'>Month {m} — no data yet</div>""", unsafe_allow_html=True)
                continue
            drift   = d.get("any_drift_detected",False)
            dtype   = d.get("actual_drift_injected","none")
            outcome = dec.get("outcome","—") if dec else "—"
            icon    = "🔴" if drift else "🟢"
            bc      = "br" if drift else "bg"
            bt      = dtype.upper().replace("_"," ") if drift else "STABLE"
            oc      = "#ffbe00" if outcome=="REJECTED" else "#00ff88" if outcome=="DEPLOYED" else "#64748b"
            st.markdown(f"""<div style='display:flex;align-items:center;gap:8px;padding:7px 11px;background:var(--card);border:1px solid var(--border);border-radius:5px;margin-bottom:5px;'>
                <span style='font-family:Space Mono,monospace;font-size:.7rem;color:#64748b;width:55px;'>Month {m}</span>
                <span>{icon}</span><span class='badge {bc}'>{bt}</span>
                <span style='margin-left:auto;font-size:.7rem;color:{oc};font-family:Space Mono,monospace;'>{outcome}</span>
            </div>""", unsafe_allow_html=True)

    if meta:
        st.markdown("<hr class='hr'><div class='sec'>Active Model — Full Metrics</div>", unsafe_allow_html=True)
        metrics_display = [
            ("Accuracy","accuracy"),("Precision","precision"),
            ("Recall","recall"),("F1","f1_score"),("AUC-ROC","auc_roc")
        ]
        cols = st.columns(5)
        for (label,key), col in zip(metrics_display, cols):
            v = meta.get("metrics",{}).get(key, 0)
            c = "#00ff88" if isinstance(v,float) and v>.95 else "#ffbe00" if isinstance(v,float) and v>.9 else "#ff3b5c"
            with col:
                st.markdown(f"""<div class='dg-card' style='text-align:center;padding:14px;'>
                    <div style='font-family:Space Mono,monospace;font-size:1.3rem;color:{c};font-weight:700;'>{v if v else '—'}</div>
                    <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;'>{label}</div>
                </div>""", unsafe_allow_html=True)


# TAB 2 — DRIFT MONITOR

with tab2:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;margin-bottom:3px;'>Drift Monitor</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.82rem;color:#64748b;margin-bottom:20px;'>PSI · KS Test · ADWIN — distribution analysis across all months</div>", unsafe_allow_html=True)

    num_feats = ["cibil_score","loan_amount","income_annum","loan_term",
                 "residential_assets","commercial_assets","luxury_assets","bank_assets","num_dependents"]

    psi_mat = []
    has_data = False
    for feat in num_feats:
        row = []
        for m in range(1,7):
            d = load_drift(m)
            val = d["psi_results"].get(feat,{}).get("psi",0) if d and "psi_results" in d else 0
            row.append(val)
            if val > 0: has_data = True
        psi_mat.append(row)

    if has_data:
        st.markdown("<div class='sec'>PSI Heatmap — All Features × All Months</div>", unsafe_allow_html=True)
        fig_h = go.Figure(go.Heatmap(
            z=psi_mat, x=[f"M{m}" for m in range(1,7)], y=num_feats,
            colorscale=[[0,"#0f1623"],[0.3,"#1e3a5f"],[0.6,"#b45309"],[1,"#ff3b5c"]],
            zmin=0, zmax=0.35,
            text=[[f"{v:.3f}" for v in row] for row in psi_mat],
            texttemplate="%{text}", textfont=dict(size=10,family="Space Mono"),
            colorbar=dict(tickfont=dict(color="#64748b"),bgcolor="rgba(0,0,0,0)",bordercolor="#1e2d45",title=dict(text="PSI",side="right"))
        ))
        pchart(fig_h, 310)
    else:
        st.info("No drift data yet. Run: `python scripts/run_drift_detection.py --all`")

    st.markdown("<hr class='hr'><div class='sec'>Per-Month Detail</div>", unsafe_allow_html=True)
    sel = st.select_slider("Select Month", options=list(range(1,7)), format_func=lambda x:f"Month {x}", label_visibility="collapsed")
    d   = load_drift(sel)
    if d:
        any_drift = d.get("any_drift_detected",False)
        dtype     = d.get("actual_drift_injected","none")
        if any_drift:
            st.markdown(f"<div class='alert-bad'>🔴 <strong>Drift Detected</strong> — {dtype.replace('_',' ').title()} · Max PSI: <strong>{d.get('max_psi',0):.4f}</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-ok'>🟢 <strong>No Drift</strong> — All feature distributions stable</div>", unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("<div class='sec'>PSI Scores</div>", unsafe_allow_html=True)
            psi = d.get("psi_results",{})
            if psi:
                fp = list(psi.keys()); vp = [psi[f]["psi"] for f in fp]
                cp = ["#ff3b5c" if v>.2 else "#ffbe00" if v>.1 else "#3b82f6" for v in vp]
                fig_p = go.Figure(go.Bar(x=vp, y=fp, orientation="h", marker_color=cp,
                    text=[f"{v:.4f}" for v in vp], textposition="outside",
                    textfont=dict(family="Space Mono",size=10)))
                fig_p.add_vline(x=0.2, line_dash="dot", line_color="#ff3b5c",
                                annotation_text="0.2", annotation_font_color="#ff3b5c", annotation_font_size=10)
                pchart(fig_p, 285, xa={"range":[0,.45]})

        with c2:
            st.markdown("<div class='sec'>KS Test p-values</div>", unsafe_allow_html=True)
            ks = d.get("ks_results",{})
            if ks:
                fk = list(ks.keys()); pk = [ks[f]["p_value"] for f in fk]
                ck = ["#ff3b5c" if p<.05 else "#00ff88" for p in pk]
                fig_k = go.Figure(go.Bar(x=pk, y=fk, orientation="h", marker_color=ck,
                    text=[f"{p:.4f}" for p in pk], textposition="outside",
                    textfont=dict(family="Space Mono",size=10)))
                fig_k.add_vline(x=0.05, line_dash="dot", line_color="#ffbe00",
                                annotation_text="p=0.05", annotation_font_color="#ffbe00", annotation_font_size=10)
                pchart(fig_k, 285, xa={"range":[0,1.15]})

        acc = d.get("concept_drift",{}).get("accuracy_check",{})
        if acc:
            st.markdown("<hr class='hr'><div class='sec'>Accuracy Analysis</div>", unsafe_allow_html=True)
            drop = acc.get("accuracy_drop",0)
            dc   = "#ff3b5c" if drop>.05 else "#ffbe00" if drop>.02 else "#00ff88"
            baseline_v = acc.get("baseline_accuracy","—")
            current_v  = acc.get("current_accuracy","—")
            ca = "#00ff88" if isinstance(current_v,float) and current_v>.95 else "#ffbe00" if isinstance(current_v,float) and current_v>.85 else "#ff3b5c"
            for col,(label,val,c) in zip(st.columns(3),[
                ("Baseline Accuracy", baseline_v, "#64748b"),
                ("Current Accuracy", current_v, ca),
                ("Accuracy Drop", f"{drop:+.4f}", dc)
            ]):
                with col:
                    st.markdown(f"""<div class='dg-card' style='text-align:center;'>
                        <div style='font-family:Space Mono,monospace;font-size:1.35rem;color:{c};font-weight:700;'>{val}</div>
                        <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;'>{label}</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.info(f"No drift data for Month {sel}. Run drift detection first.")


# TAB 3 — SHAP

with tab3:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;margin-bottom:3px;'>SHAP Explainability</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.82rem;color:#64748b;margin-bottom:20px;'>Feature importance from baseline model — why the model predicts what it predicts</div>", unsafe_allow_html=True)

    shap_meta = load_meta_for_shap()

    if shap_meta is None:
        st.error("⚠️ SHAP data not found. Make sure train_baseline.py has been run and models/metadata_v1.0.0.json exists.")
        st.info("Run: `python scripts/train_baseline.py`")
    else:
        shap = shap_meta.get("shap_importance", {})
        if not shap:
            st.error("SHAP importance data is empty in the metadata file.")
        else:
            feats = list(shap.keys())
            vals  = list(shap.values())
            maxv  = max(vals) if vals else 1

            st.markdown(f"<div style='font-size:.75rem;color:#64748b;margin-bottom:16px;'>Source: Model <span style='font-family:Space Mono,monospace;color:#00ff88;'>{shap_meta.get('version','v1.0.0')}</span> · {len(feats)} features</div>", unsafe_allow_html=True)

            c1, c2 = st.columns([3, 2])

            with c1:
                st.markdown("<div class='sec'>Mean |SHAP| Value per Feature</div>", unsafe_allow_html=True)
                bcs = []
                for i, v in enumerate(vals):
                    op = 0.35 + 0.65*(v/maxv)
                    bcs.append(f"rgba(59,130,246,{op:.2f})")
                if bcs: bcs[0] = "#00ff88"

                fig_s = go.Figure(go.Bar(
                    x=vals[::-1], y=feats[::-1], orientation="h",
                    marker_color=bcs[::-1],
                    text=[f"{v:.4f}" for v in vals[::-1]],
                    textposition="outside",
                    textfont=dict(family="Space Mono",size=11,color="#e8edf5")
                ))
                pchart(fig_s, 380, xa={"range":[0, maxv*1.35]})

            with c2:
                st.markdown("<div class='sec'>Feature Ranking</div>", unsafe_allow_html=True)
                for rank,(feat,val) in enumerate(zip(feats,vals),1):
                    pct   = val/maxv*100
                    color = "#00ff88" if rank==1 else "#3b82f6" if rank<=3 else "#64748b"
                    grad  = "linear-gradient(90deg,#00ff88,#3b82f6)" if rank==1 else "linear-gradient(90deg,#1e3a5f,#3b82f6)"
                    st.markdown(f"""
                    <div class='shap-row'>
                        <span style='font-family:Space Mono,monospace;font-size:.62rem;color:#64748b;width:20px;'>#{rank}</span>
                        <span class='shap-lbl' style='color:{color};'>{feat}</span>
                        <div class='shap-bg'><div class='shap-fil' style='width:{pct:.1f}%;background:{grad};'></div></div>
                        <span class='shap-val'>{val:.4f}</span>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<hr class='hr'><div class='sec'>Business Interpretation</div>", unsafe_allow_html=True)

            interps = {
                "cibil_score":("Critical","#ff3b5c","Dominates all other features by a large margin (4.74 vs next at 1.53). CIBIL score is India's primary credit metric — regulatory threshold for most Indian banks. When this feature drifts, model accuracy degrades fastest."),
                "loan_term":("High","#ffbe00","Second most impactful feature. Longer loan terms expose the bank to higher uncertainty and default risk. Sensitive to portfolio policy changes."),
                "residential_assets":("Medium","#3b82f6","Collateral signal — higher residential assets correlate with repayment capacity and willingness to pay."),
                "loan_amount":("Medium","#3b82f6","Larger loan exposures carry higher absolute default risk. Interacts strongly with CIBIL score and income."),
                "commercial_assets":("Medium","#3b82f6","Business asset holdings — more important for SME and corporate applicants than retail."),
                "income_annum":("Medium","#3b82f6","Annual income determines monthly repayment capacity. Highly sensitive to economic drift events."),
                "luxury_assets":("Low","#64748b","Secondary collateral signal. Notably volatile — showed the highest PSI in Month 3 data drift."),
                "bank_assets":("Low","#64748b","Liquid savings as safety net. Low SHAP value suggests the model relies more on credit history than savings."),
                "education":("Minimal","#334155","Weak predictor once financial metrics are included."),
                "num_dependents":("Minimal","#334155","Financial obligations through dependents contribute minimally."),
                "self_employed":("Minimal","#334155","Employment type has the lowest predictive contribution in this dataset."),
            }
            cols_interp = st.columns(2)
            for idx,(feat,val) in enumerate(zip(feats[:8], vals[:8])):
                interp = interps.get(feat, ("—","#64748b","No interpretation available."))
                level, color, desc = interp
                with cols_interp[idx % 2]:
                    st.markdown(f"""<div class='dg-card' style='margin-bottom:8px;padding:12px 16px;'>
                        <div style='display:flex;align-items:center;gap:9px;margin-bottom:5px;'>
                            <span style='font-family:Space Mono,monospace;font-size:.79rem;color:{color};font-weight:700;'>{feat}</span>
                            <span class='badge' style='background:rgba(0,0,0,.3);color:{color};border-color:{color}40;'>{level}</span>
                            <span style='margin-left:auto;font-family:Space Mono,monospace;font-size:.7rem;color:#64748b;'>{val:.4f}</span>
                        </div>
                        <div style='font-size:.77rem;color:#94a3b8;'>{desc}</div>
                    </div>""", unsafe_allow_html=True)


# TAB 4 — RL ANALYSIS

with tab4:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;margin-bottom:3px;'>Q-Learning Retraining Agent</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.82rem;color:#64748b;margin-bottom:20px;'>Reinforcement learning agent that decides HOW to retrain — not just whether to retrain</div>", unsafe_allow_html=True)

    # Concept explanation
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='dg-card card-b' style='text-align:center;padding:20px;'>
            <div style='font-size:2rem;margin-bottom:8px;'>🎯</div>
            <div style='font-family:Space Mono,monospace;font-size:.85rem;color:#3b82f6;font-weight:700;margin-bottom:6px;'>STATE SPACE</div>
            <div style='font-size:.78rem;color:#94a3b8;'>drift_type × accuracy_trend</div>
            <div style='font-size:.78rem;color:#94a3b8;margin-top:4px;'>= 9 possible states</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='dg-card card-g' style='text-align:center;padding:20px;'>
            <div style='font-size:2rem;margin-bottom:8px;'>⚡</div>
            <div style='font-family:Space Mono,monospace;font-size:.85rem;color:#00ff88;font-weight:700;margin-bottom:6px;'>ACTION SPACE</div>
            <div style='font-size:.78rem;color:#94a3b8;'>none · partial · full</div>
            <div style='font-size:.78rem;color:#94a3b8;margin-top:4px;'>3 retraining strategies</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='dg-card card-y' style='text-align:center;padding:20px;'>
            <div style='font-size:2rem;margin-bottom:8px;'>🏆</div>
            <div style='font-family:Space Mono,monospace;font-size:.85rem;color:#ffbe00;font-weight:700;margin-bottom:6px;'>REWARD</div>
            <div style='font-size:.78rem;color:#94a3b8;'>new_accuracy − old_accuracy</div>
            <div style='font-size:.78rem;color:#94a3b8;margin-top:4px;'>Positive = good decision</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)

    # Q-Table Visualization
    q_table = load_q_table()

    if not q_table:
        st.info("Q-table not found. Run retraining pipeline first: `python scripts/retraining_pipeline.py --all`")
    else:
        st.markdown("<div class='sec'>Q-Table — Learned Values (higher = preferred action)</div>", unsafe_allow_html=True)

        states  = list(q_table.keys())
        actions = ["none", "partial", "full"]

        # Build heatmap data
        z_vals   = []
        y_labels = []
        for state in states:
            row = [q_table[state].get(a, 0) for a in actions]
            z_vals.append(row)
            # Make state labels readable
            parts = state.split("|")
            drift_label = parts[0].replace("_"," ").title() if parts else state
            trend_label = parts[1].title() if len(parts)>1 else ""
            y_labels.append(f"{drift_label} / {trend_label}")

        fig_q = go.Figure(go.Heatmap(
            z=z_vals,
            x=["None\n(Keep Model)", "Partial\n(Sliding Window)", "Full\n(All Data)"],
            y=y_labels,
            colorscale=[
                [0.0, "#ff3b5c"],
                [0.4, "#0f1623"],
                [0.6, "#0f1623"],
                [1.0, "#00ff88"]
            ],
            text=[[f"{v:.4f}" for v in row] for row in z_vals],
            texttemplate="%{text}",
            textfont=dict(size=12, family="Space Mono"),
            zmid=0,
            colorbar=dict(
                tickfont=dict(color="#64748b"),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#1e2d45",
                title=dict(text="Q-Value", side="right")
            )
        ))
        pchart(fig_q, 300 + len(states)*40)

        # Q-Table Details
        st.markdown("<hr class='hr'><div class='sec'>Q-Table Detail — Best Action per State</div>", unsafe_allow_html=True)
        tbl = "<table class='dtable'><thead><tr><th>State</th><th>Best Action</th><th>None Q-Val</th><th>Partial Q-Val</th><th>Full Q-Val</th><th>Confidence</th></tr></thead><tbody>"
        for state, av in q_table.items():
            best_action = max(av, key=av.get)
            best_val    = av[best_action]
            vals_list   = list(av.values())
            all_equal   = len(set(round(v,4) for v in vals_list)) == 1
            confidence  = "Learning..." if all_equal else ("High" if abs(best_val) > 0.05 else "Low")
            conf_color  = "#64748b" if all_equal else ("#00ff88" if confidence=="High" else "#ffbe00")
            parts = state.split("|")
            state_disp = f"{parts[0].replace('_',' ')} / {parts[1] if len(parts)>1 else ''}"

            none_v    = av.get("none", 0)
            partial_v = av.get("partial", 0)
            full_v    = av.get("full", 0)

            def fmt_q(v, is_best):
                c = "#00ff88" if is_best else ("#ff3b5c" if v < 0 else "#64748b")
                bold = "font-weight:700;" if is_best else ""
                return f"<span style='color:{c};{bold}font-family:Space Mono,monospace;'>{v:.4f}</span>"

            tbl += f"""<tr>
                <td style='font-size:.78rem;'>{state_disp}</td>
                <td><span class='badge {"bg" if best_action=="none" else "bb" if best_action=="partial" else "by"}'>{best_action.upper()}</span></td>
                <td>{fmt_q(none_v, best_action=="none")}</td>
                <td>{fmt_q(partial_v, best_action=="partial")}</td>
                <td>{fmt_q(full_v, best_action=="full")}</td>
                <td style='color:{conf_color};font-size:.75rem;'>{confidence}</td>
            </tr>"""
        tbl += "</tbody></table>"
        st.markdown(f"<div class='dg-card'>{tbl}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)

    # Retraining History
    st.markdown("<div class='sec'>Retraining Episode History</div>", unsafe_allow_html=True)
    retrain_history = []
    for m in range(1,7):
        r = load_retrain(m)
        if r:
            retrain_history.append(r)

    if retrain_history:
        # Reward chart
        months_rl = [f"Month {r['month']}" for r in retrain_history]
        rewards   = [r.get("reward",0) for r in retrain_history]
        actions_rl= [r.get("rl_action","—") for r in retrain_history]
        colors_rl = ["#00ff88" if rw>0 else "#ff3b5c" if rw<0 else "#64748b" for rw in rewards]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='sec'>Reward per Episode</div>", unsafe_allow_html=True)
            fig_r = go.Figure(go.Bar(
                x=months_rl, y=rewards, marker_color=colors_rl,
                text=[f"{rw:+.4f}" for rw in rewards],
                textposition="outside", textfont=dict(family="Space Mono",size=11)
            ))
            fig_r.add_hline(y=0, line_color="#64748b", line_width=1)
            pchart(fig_r, 250)

        with c2:
            st.markdown("<div class='sec'>Action Selected per Episode</div>", unsafe_allow_html=True)
            action_colors = {"none":"#64748b","partial":"#3b82f6","full":"#ffbe00"}
            ac = [action_colors.get(a,"#64748b") for a in actions_rl]
            fig_a = go.Figure(go.Bar(
                x=months_rl, y=[1]*len(months_rl),
                marker_color=ac,
                text=[a.upper() for a in actions_rl],
                textposition="inside",
                textfont=dict(family="Space Mono", size=12, color="#e8edf5")
            ))
            fig_a.update_layout(**BL, height=250, showlegend=False,
                                yaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
                                xaxis=XA)
            st.plotly_chart(fig_a, use_container_width=True)

        # Episode detail table
        st.markdown("<div class='sec'>Episode Detail</div>", unsafe_allow_html=True)
        etbl = "<table class='dtable'><thead><tr><th>Month</th><th>Drift Type</th><th>RL State</th><th>Action</th><th>Basis</th><th>Prev Acc</th><th>New Acc</th><th>Reward</th><th>Deployed</th></tr></thead><tbody>"
        for r in retrain_history:
            rw = r.get("reward",0)
            rc = "#00ff88" if rw>0 else "#ff3b5c" if rw<0 else "#64748b"
            ac_badge = r.get("rl_action","—")
            ac_class = "bg" if ac_badge=="none" else "bb" if ac_badge=="partial" else "by"
            dep = "✅ YES" if r.get("deployed") else "❌ NO"
            dp_c = "#00ff88" if r.get("deployed") else "#ff3b5c"
            etbl += f"""<tr>
                <td style='font-family:Space Mono,monospace;color:#3b82f6;'>Month {r.get('month','—')}</td>
                <td style='font-size:.75rem;'>{r.get('drift_type','—').replace('_',' ')}</td>
                <td style='font-size:.72rem;color:#94a3b8;font-family:Space Mono,monospace;'>{r.get('rl_state','—')}</td>
                <td><span class='badge {ac_class}'>{ac_badge.upper()}</span></td>
                <td style='font-size:.72rem;color:#64748b;'>{"Explore" if "Explor" in str(r.get("rl_action","")) else "Exploit"}</td>
                <td style='font-family:Space Mono,monospace;'>{r.get('prev_accuracy','—')}</td>
                <td style='font-family:Space Mono,monospace;'>{r.get('new_accuracy','—')}</td>
                <td style='font-family:Space Mono,monospace;color:{rc};'>{rw:+.4f}</td>
                <td style='color:{dp_c};'>{dep}</td>
            </tr>"""
        etbl += "</tbody></table>"
        st.markdown(f"<div class='dg-card'>{etbl}</div>", unsafe_allow_html=True)
    else:
        st.info("No retraining episodes yet. Run: `python scripts/retraining_pipeline.py --all`")

    # Bellman equation explanation
    st.markdown("<hr class='hr'><div class='sec'>How the Agent Learns — Bellman Equation</div>", unsafe_allow_html=True)
    st.markdown("""<div class='dg-card card-b'>
        <div style='font-family:Space Mono,monospace;font-size:.95rem;color:#3b82f6;text-align:center;padding:12px 0;'>
            Q[s][a] ← Q[s][a] + α × (r + γ × max(Q[s']) − Q[s][a])
        </div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px;'>
            <div style='text-align:center;'>
                <div style='font-family:Space Mono,monospace;color:#00ff88;font-size:.9rem;font-weight:700;'>α = 0.1</div>
                <div style='font-size:.72rem;color:#64748b;margin-top:4px;'>Learning Rate<br>How fast new info overwrites old</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-family:Space Mono,monospace;color:#ffbe00;font-size:.9rem;font-weight:700;'>γ = 0.9</div>
                <div style='font-size:.72rem;color:#64748b;margin-top:4px;'>Discount Factor<br>Future rewards vs immediate</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-family:Space Mono,monospace;color:#3b82f6;font-size:.9rem;font-weight:700;'>ε = 0.2</div>
                <div style='font-size:.72rem;color:#64748b;margin-top:4px;'>Exploration Rate<br>20% random, 80% optimal</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-family:Space Mono,monospace;color:#ff3b5c;font-size:.9rem;font-weight:700;'>r = Δacc</div>
                <div style='font-size:.72rem;color:#64748b;margin-top:4px;'>Reward Signal<br>Accuracy improvement</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# TAB 5 — DECISION LOG

with tab5:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;margin-bottom:3px;'>Decision Log</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.82rem;color:#64748b;margin-bottom:20px;'>Complete audit trail — every retrain decision with full reasoning</div>", unsafe_allow_html=True)

    rows = []
    for m in range(1,7):
        dec = load_dec(m); d = load_drift(m)
        if dec:
            rows.append({"month":m,"drift":dec.get("drift_detected",False),
                "retrain":dec.get("retrain_decision",False),
                "deploy":dec.get("deploy_decision"),
                "outcome":dec.get("outcome","—"),
                "reason":dec.get("retrain_reason","—"),
                "max_psi":d.get("max_psi",0) if d else 0})

    if rows:
        nd2,nr2 = sum(1 for r in rows if r["drift"]),sum(1 for r in rows if r["retrain"])
        nD,nR   = sum(1 for r in rows if r["outcome"]=="DEPLOYED"),sum(1 for r in rows if r["outcome"]=="REJECTED")
        st.markdown(f"""<div class='mg'>
            <div class='mc card-r'><div class='mv red'>{nd2}</div><div class='ml'>Drift Events</div></div>
            <div class='mc card-y'><div class='mv yellow'>{nr2}</div><div class='ml'>Retrains</div></div>
            <div class='mc card-g'><div class='mv green'>{nD}</div><div class='ml'>Deployed</div></div>
            <div class='mc card-b'><div class='mv blue'>{nR}</div><div class='ml'>Rejected</div></div>
        </div>""", unsafe_allow_html=True)

        oc_map = {
            "NO_ACTION":"<span class='badge bg'>NO ACTION</span>",
            "DEPLOYED":"<span class='badge bb'>DEPLOYED</span>",
            "REJECTED":"<span class='badge by'>REJECTED</span>",
            "RETRAIN_FAILED":"<span class='badge br'>FAILED</span>",
            "—":"<span style='color:#64748b;font-size:.75rem;'>—</span>"
        }
        tbl = "<table class='dtable'><thead><tr><th>Month</th><th>Drift</th><th>Max PSI</th><th>Retrain</th><th>Deploy</th><th>Outcome</th><th>Reason</th></tr></thead><tbody>"
        for r in rows:
            pc = "#ff3b5c" if r["max_psi"]>.2 else "#ffbe00" if r["max_psi"]>.1 else "#64748b"
            tbl += f"""<tr>
                <td style='font-family:Space Mono,monospace;color:#3b82f6;'>Month {r['month']}</td>
                <td>{'🔴 Yes' if r['drift'] else '🟢 No'}</td>
                <td style='font-family:Space Mono,monospace;color:{pc};'>{r['max_psi']:.4f}</td>
                <td>{'🔴 Yes' if r['retrain'] else '🟢 No'}</td>
                <td>{'✅ Yes' if r['deploy'] is True else '❌ No' if r['deploy'] is False else '—'}</td>
                <td>{oc_map.get(r['outcome'],r['outcome'])}</td>
                <td style='color:#94a3b8;font-size:.76rem;max-width:260px;'>{r['reason'][:70]}{'...' if len(r['reason'])>70 else ''}</td>
            </tr>"""
        tbl += "</tbody></table>"
        st.markdown(f"<div class='dg-card'>{tbl}</div>", unsafe_allow_html=True)

        st.markdown("<hr class='hr'><div class='sec'>Detailed View</div>", unsafe_allow_html=True)
        sel = st.selectbox("Select Month", [f"Month {m}" for m in range(1,7)], label_visibility="collapsed")
        dec2 = load_dec(int(sel.split()[-1]))
        if dec2:
            c1,c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class='dg-card card-b'>
                    <div class='sec'>Decision Chain</div>
                    <div style='font-size:.83rem;line-height:2.1;'>
                        <div>📍 <b>Drift:</b> {'Yes' if dec2.get('drift_detected') else 'No'}</div>
                        <div>🔄 <b>Retrain:</b> {'Yes' if dec2.get('retrain_decision') else 'No'}</div>
                        <div>🚀 <b>Deploy:</b> {str(dec2.get('deploy_decision'))}</div>
                        <div>📊 <b>Model:</b> <span style='font-family:Space Mono,monospace;color:#00ff88;'>{dec2.get('active_model','—')}</span></div>
                    </div></div>""", unsafe_allow_html=True)
            with c2:
                oc = dec2.get("outcome","—")
                oc_c = "#00ff88" if oc=="DEPLOYED" else "#ffbe00" if oc=="REJECTED" else "#3b82f6"
                st.markdown(f"""<div class='dg-card card-{"g" if oc=="DEPLOYED" else "y" if oc=="REJECTED" else "b"}'>
                    <div class='sec'>Final Outcome</div>
                    <div style='font-family:Space Mono,monospace;font-size:1.4rem;color:{oc_c};margin:7px 0;'>{oc}</div>
                    <div style='font-size:.77rem;color:#94a3b8;'>{dec2.get("retrain_reason","—")}</div>
                    {f'<div style="font-size:.73rem;color:#64748b;margin-top:5px;">{dec2.get("deploy_reason","")}</div>' if dec2.get("deploy_reason") else ""}
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No decisions yet. Run: `python scripts/decision_engine.py --all`")


# TAB 6 — DATA EXPLORER

with tab6:
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:1.4rem;font-weight:700;margin-bottom:3px;'>Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.82rem;color:#64748b;margin-bottom:20px;'>Explore monthly batch distributions and compare against baseline</div>", unsafe_allow_html=True)

    df = load_df()
    if df is None:
        st.warning("Dataset not found. Run `python scripts/prepare_real_data.py` first.")
    else:
        c1,c2 = st.columns(2)
        with c1: feat = st.selectbox("Feature",["cibil_score","loan_amount","income_annum","loan_term","residential_assets","luxury_assets","num_dependents"])
        with c2: cm = st.selectbox("Compare Month",[3,4,5,6],format_func=lambda x:f"Month {x} vs Baseline")

        baseline = df[df["month"].isin([1,2])][feat]
        current  = df[df["month"]==cm][feat]
        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(x=baseline,name="Baseline (M1+M2)",marker_color="rgba(59,130,246,.6)",nbinsx=30,histnorm="probability density"))
        fig_d.add_trace(go.Histogram(x=current,name=f"Month {cm}",marker_color="rgba(255,59,92,.6)",nbinsx=30,histnorm="probability density"))
        fig_d.update_layout(**BL,height=295,barmode="overlay",title=dict(text=f"Distribution: {feat}",font=dict(size=13,color="#e8edf5")))
        fig_d.update_xaxes(**XA); fig_d.update_yaxes(**YA)
        st.plotly_chart(fig_d,use_container_width=True)

        st.markdown("<hr class='hr'><div class='sec'>Default Rate by Month</div>", unsafe_allow_html=True)
        md = df.groupby("month").agg(default_rate=("default_label","mean"),drift=("drift_type","first")).reset_index()
        fig_dr = go.Figure(go.Bar(
            x=[f"M{m}" for m in md["month"]], y=md["default_rate"],
            marker_color=["#ff3b5c" if d!="none" else "#3b82f6" for d in md["drift"]],
            text=[f"{r:.1%}" for r in md["default_rate"]],
            textposition="outside", textfont=dict(family="Space Mono",size=11)
        ))
        fig_dr.update_layout(**BL,height=245)
        fig_dr.update_xaxes(**XA)
        fig_dr.update_yaxes(**YA,tickformat=".0%")
        st.plotly_chart(fig_dr,use_container_width=True)

        st.markdown("<div class='sec'>Sample Data</div>", unsafe_allow_html=True)
        sm = st.select_slider("Month",options=list(range(1,7)),format_func=lambda x:f"Month {x}",label_visibility="collapsed")
        st.dataframe(df[df["month"]==sm].head(12),use_container_width=True,hide_index=True)