"""
DriftGuard — Professional Monitoring Dashboard
Step 8: Streamlit Dashboard

Run:
    streamlit run dashboard/app.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR  = os.path.join(BASE_DIR, "logs")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DriftGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:     #0a0e17;
    --card:   #0f1623;
    --raised: #161e2e;
    --border: #1e2d45;
    --green:  #00ff88;
    --red:    #ff3b5c;
    --yellow: #ffbe00;
    --blue:   #3b82f6;
    --muted:  #64748b;
    --text:   #e8edf5;
}
.stApp { background: var(--bg); }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text); }
[data-testid="stSidebar"] { background: var(--card) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; }

.dg-title { font-family:'Space Mono',monospace; font-size:1.5rem; font-weight:700; color:var(--text); margin-bottom:3px; }
.dg-sub   { font-size:0.82rem; color:var(--muted); margin-bottom:24px; }

.mg { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:22px; }
.mc { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:18px; text-align:center; }
.mv { font-family:'Space Mono',monospace; font-size:1.8rem; font-weight:700; line-height:1; margin-bottom:5px; }
.ml { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; }
.green  { color:var(--green);  } .red    { color:var(--red);    }
.yellow { color:var(--yellow); } .blue   { color:var(--blue);   }
.white  { color:var(--text);   }

.card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:18px 22px; margin-bottom:14px; }
.card-g { border-left:3px solid var(--green);  }
.card-r { border-left:3px solid var(--red);    }
.card-y { border-left:3px solid var(--yellow); }
.card-b { border-left:3px solid var(--blue);   }

.sec { font-family:'Space Mono',monospace; font-size:0.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.12em; margin-bottom:10px; }
.hr  { border:none; border-top:1px solid var(--border); margin:18px 0; }

.badge { display:inline-block; padding:2px 9px; border-radius:3px; font-family:'Space Mono',monospace; font-size:.68rem; font-weight:700; letter-spacing:.04em; }
.bg { background:rgba(0,255,136,.1); color:var(--green); border:1px solid rgba(0,255,136,.3); }
.br { background:rgba(255,59,92,.1); color:var(--red);   border:1px solid rgba(255,59,92,.3); }
.by { background:rgba(255,190,0,.1); color:var(--yellow);border:1px solid rgba(255,190,0,.3); }
.bb { background:rgba(59,130,246,.1);color:var(--blue);  border:1px solid rgba(59,130,246,.3);}

.dtable { width:100%; border-collapse:collapse; font-size:.83rem; }
.dtable th { font-family:'Space Mono',monospace; font-size:.65rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; }
.dtable td { padding:10px 12px; border-bottom:1px solid rgba(30,45,69,.4); }
.dtable tr:hover td { background:rgba(255,255,255,.02); }

.shap-row { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.shap-lbl { font-size:.8rem; color:var(--text); width:155px; flex-shrink:0; }
.shap-bg  { flex:1; background:var(--raised); border-radius:3px; height:7px; overflow:hidden; }
.shap-fil { height:100%; border-radius:3px; }
.shap-val { font-family:'Space Mono',monospace; font-size:.72rem; color:var(--muted); width:50px; text-align:right; }

.alert-ok  { background:rgba(0,255,136,.06); border:1px solid rgba(0,255,136,.2); border-radius:6px; padding:11px 15px; margin-bottom:12px; font-size:.84rem; }
.alert-bad { background:rgba(255,59,92,.07);  border:1px solid rgba(255,59,92,.25); border-radius:6px; padding:11px 15px; margin-bottom:12px; font-size:.84rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════
@st.cache_data
def load_drift(month):
    p = os.path.join(LOGS_DIR, f"drift_month_{month}.json")
    return json.load(open(p)) if os.path.exists(p) else None

@st.cache_data
def load_decision(month):
    p = os.path.join(LOGS_DIR, f"decision_month_{month}.json")
    return json.load(open(p)) if os.path.exists(p) else None

@st.cache_data
def load_summary():
    p = os.path.join(LOGS_DIR, "drift_summary.json")
    return json.load(open(p)) if os.path.exists(p) else []

@st.cache_data
def load_active_meta():
    for f in sorted(os.listdir(MODEL_DIR), reverse=True):
        if f.startswith("metadata_") and f.endswith(".json"):
            m = json.load(open(os.path.join(MODEL_DIR, f)))
            if m.get("status") == "active":
                return m
    return None

@st.cache_data
def load_dataset():
    p = os.path.join(DATA_DIR, "full_dataset.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e8edf5", size=12),
    margin=dict(l=12, r=12, t=28, b=12),
    xaxis=dict(showgrid=False, zeroline=False, color="#64748b", linecolor="#1e2d45", tickcolor="#1e2d45"),
    yaxis=dict(showgrid=True, gridcolor="#1e2d45", zeroline=False, color="#64748b", linecolor="#1e2d45"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d45")
)


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:6px 0 18px 0;'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;color:#00ff88;'>🛡️ DriftGuard</div>
        <div style='font-size:.7rem;color:#64748b;margin-top:3px;'>Financial ML Monitor</div>
    </div>""", unsafe_allow_html=True)

    page = st.selectbox("Nav", ["📊 Overview","🔍 Drift Monitor","🧠 SHAP","📋 Decision Log","📁 Data Explorer"],
                        label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e2d45;margin:14px 0;'>", unsafe_allow_html=True)
    meta = load_active_meta()
    if meta:
        st.markdown(f"""
        <div style='font-size:.78rem;'>
            <div style='color:#64748b;margin-bottom:4px;'>ACTIVE MODEL</div>
            <div style='font-family:Space Mono,monospace;color:#00ff88;font-size:.9rem;margin-bottom:12px;'>{meta['version']}</div>
            <div style='color:#64748b;margin-bottom:4px;'>AUC-ROC</div>
            <div style='font-family:Space Mono,monospace;color:#e8edf5;margin-bottom:12px;'>{meta['metrics']['auc_roc']}</div>
            <div style='color:#64748b;margin-bottom:4px;'>ACCURACY</div>
            <div style='font-family:Space Mono,monospace;color:#e8edf5;'>{meta['metrics']['accuracy']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d45;margin:14px 0;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.68rem;color:#64748b;'>Updated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("<div class='dg-title'>System Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='dg-sub'>DriftGuard · Adaptive Drift Detection & Retraining for Financial ML Systems</div>", unsafe_allow_html=True)

    summary  = load_summary()
    meta     = load_active_meta()
    n_drift  = sum(1 for r in summary if r.get("any_drift_detected"))
    n_retrain= sum(1 for r in summary if r.get("trigger_retrain"))

    st.markdown(f"""
    <div class='mg'>
        <div class='mc card-g'><div class='mv green'>{meta['metrics']['accuracy'] if meta else '—'}</div><div class='ml'>Accuracy</div></div>
        <div class='mc card-b'><div class='mv blue'>{meta['metrics']['auc_roc'] if meta else '—'}</div><div class='ml'>AUC-ROC</div></div>
        <div class='mc card-{'r' if n_drift else 'g'}'><div class='mv {'red' if n_drift else 'green'}'>{n_drift}</div><div class='ml'>Drift Events</div></div>
        <div class='mc card-y'><div class='mv yellow'>{n_retrain}</div><div class='ml'>Retrain Triggers</div></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='sec'>Accuracy Trend — Monthly</div>", unsafe_allow_html=True)
        months_l, accs, flags = [], [], []
        for m in range(1, 7):
            d = load_drift(m)
            if d:
                months_l.append(f"M{m}")
                acc = d.get("concept_drift", {}).get("accuracy_check", {}).get("current_accuracy")
                accs.append(acc if acc else (meta["metrics"]["accuracy"] if meta else 0.97))
                flags.append(d.get("any_drift_detected", False))

        fig = go.Figure()
        for i, flag in enumerate(flags):
            if flag:
                fig.add_vrect(x0=i-.4, x1=i+.4, fillcolor="rgba(255,59,92,.07)", line_width=0, layer="below")
        fig.add_trace(go.Scatter(
            x=months_l, y=accs, mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=9, color=["#ff3b5c" if f else "#00ff88" for f in flags],
                        line=dict(width=2, color="#0f1623")),
            fill="tozeroy", fillcolor="rgba(59,130,246,.04)", name="Accuracy"
        ))
        if meta:
            fig.add_hline(y=meta["metrics"]["accuracy"], line_dash="dot",
                          line_color="rgba(0,255,136,.35)",
                          annotation_text="Baseline", annotation_font_color="#00ff88",
                          annotation_font_size=10)
        fig.update_layout(**LAYOUT, height=250, yaxis=dict(**LAYOUT["yaxis"], range=[.75, 1.02]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sec'>Drift Timeline</div>", unsafe_allow_html=True)
        for m in range(1, 7):
            d   = load_drift(m)
            dec = load_decision(m)
            if not d: continue
            drift    = d.get("any_drift_detected", False)
            dtype    = d.get("actual_drift_injected", "none")
            outcome  = dec.get("outcome", "—") if dec else "—"
            icon     = "🔴" if drift else "🟢"
            badge_c  = "br" if drift else "bg"
            badge_t  = dtype.upper().replace("_"," ") if drift else "STABLE"
            oc_color = "#ffbe00" if outcome == "REJECTED" else "#00ff88" if outcome == "DEPLOYED" else "#64748b"
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:8px;padding:7px 11px;
                        background:var(--card);border:1px solid var(--border);
                        border-radius:5px;margin-bottom:5px;'>
                <span style='font-family:Space Mono,monospace;font-size:.7rem;color:#64748b;width:55px;'>Month {m}</span>
                <span>{icon}</span>
                <span class='badge {badge_c}'>{badge_t}</span>
                <span style='margin-left:auto;font-size:.7rem;color:{oc_color};font-family:Space Mono,monospace;'>{outcome}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>Active Model — Full Metrics</div>", unsafe_allow_html=True)
    if meta:
        cols = st.columns(5)
        for (label, key), col in zip([
            ("Accuracy","accuracy"),("Precision","precision"),
            ("Recall","recall"),("F1 Score","f1_score"),("AUC-ROC","auc_roc")
        ], cols):
            v = meta["metrics"][key]
            c = "#00ff88" if v>.95 else "#ffbe00" if v>.9 else "#ff3b5c"
            with col:
                st.markdown(f"""<div class='card' style='text-align:center;padding:14px;'>
                    <div style='font-family:Space Mono,monospace;font-size:1.35rem;color:{c};font-weight:700;'>{v}</div>
                    <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;'>{label}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — DRIFT MONITOR
# ═══════════════════════════════════════════════════════
elif page == "🔍 Drift Monitor":
    st.markdown("<div class='dg-title'>Drift Monitor</div>", unsafe_allow_html=True)
    st.markdown("<div class='dg-sub'>PSI · KS Test · ADWIN — distribution analysis across all months</div>", unsafe_allow_html=True)

    num_feats = ["cibil_score","loan_amount","income_annum","loan_term",
                 "residential_assets","commercial_assets","luxury_assets","bank_assets","num_dependents"]

    # PSI heatmap
    st.markdown("<div class='sec'>PSI Heatmap — All Features × All Months</div>", unsafe_allow_html=True)
    psi_mat = []
    for feat in num_feats:
        row = []
        for m in range(1, 7):
            d = load_drift(m)
            row.append(d["psi_results"][feat]["psi"] if d and feat in d.get("psi_results",{}) else 0)
        psi_mat.append(row)

    fig_h = go.Figure(go.Heatmap(
        z=psi_mat, x=[f"M{m}" for m in range(1,7)], y=num_feats,
        colorscale=[[0,"#0f1623"],[0.3,"#1e3a5f"],[0.6,"#b45309"],[1,"#ff3b5c"]],
        zmin=0, zmax=0.35,
        text=[[f"{v:.3f}" for v in row] for row in psi_mat],
        texttemplate="%{text}", textfont=dict(size=10, family="Space Mono"),
        colorbar=dict(tickfont=dict(color="#64748b"), bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d45",
                      title=dict(text="PSI", side="right"))
    ))
    fig_h.update_layout(**LAYOUT, height=320)
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.markdown("<div class='sec'>Detailed Analysis — Select Month</div>", unsafe_allow_html=True)

    sel = st.select_slider("Month", options=list(range(1,7)), format_func=lambda x: f"Month {x}", label_visibility="collapsed")
    d   = load_drift(sel)

    if d:
        any_drift = d.get("any_drift_detected", False)
        dtype     = d.get("actual_drift_injected","none")

        if any_drift:
            st.markdown(f"<div class='alert-bad'>🔴 <strong>Drift Detected</strong> — {dtype.replace('_',' ').title()} &nbsp;·&nbsp; Max PSI: <strong>{d.get('max_psi',0):.4f}</strong></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-ok'>🟢 <strong>No Drift</strong> — All feature distributions stable</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='sec'>PSI Scores</div>", unsafe_allow_html=True)
            psi = d.get("psi_results", {})
            if psi:
                feats_p = list(psi.keys())
                vals_p  = [psi[f]["psi"] for f in feats_p]
                cols_p  = ["#ff3b5c" if v>.2 else "#ffbe00" if v>.1 else "#3b82f6" for v in vals_p]
                fig_p   = go.Figure(go.Bar(x=vals_p, y=feats_p, orientation="h",
                    marker_color=cols_p, text=[f"{v:.4f}" for v in vals_p],
                    textposition="outside", textfont=dict(family="Space Mono", size=10)))
                fig_p.add_vline(x=0.2, line_dash="dot", line_color="#ff3b5c",
                                annotation_text="0.2", annotation_font_color="#ff3b5c", annotation_font_size=10)
                fig_p.update_layout(**LAYOUT, height=290, xaxis=dict(**LAYOUT["xaxis"], range=[0,.42]))
                st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            st.markdown("<div class='sec'>KS Test p-values</div>", unsafe_allow_html=True)
            ks = d.get("ks_results", {})
            if ks:
                feats_k = list(ks.keys())
                pvals   = [ks[f]["p_value"] for f in feats_k]
                cols_k  = ["#ff3b5c" if p<.05 else "#00ff88" for p in pvals]
                fig_k   = go.Figure(go.Bar(x=pvals, y=feats_k, orientation="h",
                    marker_color=cols_k, text=[f"{p:.4f}" for p in pvals],
                    textposition="outside", textfont=dict(family="Space Mono", size=10)))
                fig_k.add_vline(x=0.05, line_dash="dot", line_color="#ffbe00",
                                annotation_text="p=0.05", annotation_font_color="#ffbe00", annotation_font_size=10)
                fig_k.update_layout(**LAYOUT, height=290, xaxis=dict(**LAYOUT["xaxis"], range=[0, 1.15]))
                st.plotly_chart(fig_k, use_container_width=True)

        acc = d.get("concept_drift",{}).get("accuracy_check",{})
        if acc:
            st.markdown("<hr class='hr'><div class='sec'>Accuracy Analysis</div>", unsafe_allow_html=True)
            drop = acc.get("accuracy_drop",0)
            color_drop = "#ff3b5c" if drop>.05 else "#ffbe00" if drop>.02 else "#00ff88"
            c1,c2,c3 = st.columns(3)
            for col, label, val, c in [
                (c1,"Baseline Accuracy", acc.get("baseline_accuracy","—"), "#64748b"),
                (c2,"Current Accuracy",  acc.get("current_accuracy","—"),
                    "#00ff88" if isinstance(acc.get("current_accuracy"),float) and acc["current_accuracy"]>.95
                    else "#ffbe00" if isinstance(acc.get("current_accuracy"),float) and acc["current_accuracy"]>.85 else "#ff3b5c"),
                (c3,"Accuracy Drop", f"{drop:+.4f}", color_drop)
            ]:
                with col:
                    st.markdown(f"""<div class='card' style='text-align:center;'>
                        <div style='font-family:Space Mono,monospace;font-size:1.4rem;color:{c};font-weight:700;'>{val}</div>
                        <div style='font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;'>{label}</div>
                    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 3 — SHAP
# ═══════════════════════════════════════════════════════
elif page == "🧠 SHAP":
    st.markdown("<div class='dg-title'>SHAP Explainability</div>", unsafe_allow_html=True)
    st.markdown("<div class='dg-sub'>Understanding why the model predicts what it predicts — feature importance analysis</div>", unsafe_allow_html=True)

    meta = load_active_meta()
    if meta and "shap_importance" in meta:
        shap  = meta["shap_importance"]
        feats = list(shap.keys())
        vals  = list(shap.values())
        maxv  = max(vals)

        c1, c2 = st.columns([3,2])

        with c1:
            st.markdown("<div class='sec'>Mean |SHAP| Value per Feature</div>", unsafe_allow_html=True)
            bar_colors = []
            for i, v in enumerate(vals):
                op = 0.35 + 0.65*(v/maxv)
                bar_colors.append(f"rgba(59,130,246,{op:.2f})")
            bar_colors[0] = "#00ff88"

            fig_s = go.Figure(go.Bar(
                x=vals[::-1], y=feats[::-1], orientation="h",
                marker_color=bar_colors[::-1],
                text=[f"{v:.4f}" for v in vals[::-1]],
                textposition="outside",
                textfont=dict(family="Space Mono", size=11, color="#e8edf5")
            ))
            fig_s.update_layout(**LAYOUT, height=370, xaxis=dict(**LAYOUT["xaxis"], range=[0, maxv*1.35]))
            st.plotly_chart(fig_s, use_container_width=True)

        with c2:
            st.markdown("<div class='sec'>Feature Ranking</div>", unsafe_allow_html=True)
            for rank, (feat, val) in enumerate(zip(feats, vals), 1):
                pct   = val/maxv*100
                color = "#00ff88" if rank==1 else "#3b82f6" if rank<=3 else "#64748b"
                grad  = "linear-gradient(90deg,#00ff88,#3b82f6)" if rank==1 else "linear-gradient(90deg,#1e3a5f,#3b82f6)"
                st.markdown(f"""
                <div class='shap-row'>
                    <span style='font-family:Space Mono,monospace;font-size:.62rem;color:#64748b;width:18px;'>#{rank}</span>
                    <span class='shap-lbl' style='color:{color};'>{feat}</span>
                    <div class='shap-bg'><div class='shap-fil' style='width:{pct:.1f}%;background:{grad};'></div></div>
                    <span class='shap-val'>{val:.4f}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='hr'>", unsafe_allow_html=True)
        st.markdown("<div class='sec'>Feature Interpretation</div>", unsafe_allow_html=True)

        interps = {
            "cibil_score":        ("Critical","#ff3b5c","Dominant predictor. CIBIL score alone drives most decisions. A shift here causes the fastest model degradation."),
            "loan_term":          ("High",    "#ffbe00","Longer terms increase default risk. Second most impactful feature — sensitive to portfolio policy changes."),
            "residential_assets": ("Medium",  "#3b82f6","Acts as collateral signal. Higher value correlates with repayment capacity."),
            "loan_amount":        ("Medium",  "#3b82f6","Larger loans carry higher default risk. Interacts strongly with income and assets."),
            "commercial_assets":  ("Medium",  "#3b82f6","Business asset holdings — important for corporate and SME loan decisions."),
            "income_annum":       ("Medium",  "#3b82f6","Annual income determines repayment capacity. Sensitive to economic drift events."),
        }
        for feat, val in zip(feats[:6], vals[:6]):
            if feat in interps:
                level, color, desc = interps[feat]
                st.markdown(f"""
                <div class='card' style='margin-bottom:7px;padding:13px 17px;'>
                    <div style='display:flex;align-items:center;gap:10px;margin-bottom:5px;'>
                        <span style='font-family:Space Mono,monospace;font-size:.8rem;color:{color};font-weight:700;'>{feat}</span>
                        <span class='badge' style='background:rgba(0,0,0,.3);color:{color};border-color:{color}40;'>{level}</span>
                        <span style='margin-left:auto;font-family:Space Mono,monospace;font-size:.72rem;color:#64748b;'>{val:.4f}</span>
                    </div>
                    <div style='font-size:.78rem;color:#94a3b8;'>{desc}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 4 — DECISION LOG
# ═══════════════════════════════════════════════════════
elif page == "📋 Decision Log":
    st.markdown("<div class='dg-title'>Decision Log</div>", unsafe_allow_html=True)
    st.markdown("<div class='dg-sub'>Complete audit trail — every retrain decision with full reasoning</div>", unsafe_allow_html=True)

    rows = []
    for m in range(1,7):
        dec = load_decision(m)
        d   = load_drift(m)
        if dec:
            rows.append({
                "month":   m,
                "drift":   dec.get("drift_detected", False),
                "retrain": dec.get("retrain_decision", False),
                "deploy":  dec.get("deploy_decision"),
                "outcome": dec.get("outcome","—"),
                "reason":  dec.get("retrain_reason","—"),
                "model":   dec.get("new_version") if dec.get("outcome")=="DEPLOYED" else dec.get("active_model"),
                "max_psi": d.get("max_psi",0) if d else 0
            })

    nd = sum(1 for r in rows if r["drift"])
    nr = sum(1 for r in rows if r["retrain"])
    nD = sum(1 for r in rows if r["outcome"]=="DEPLOYED")
    nR = sum(1 for r in rows if r["outcome"]=="REJECTED")

    st.markdown(f"""
    <div class='mg'>
        <div class='mc card-r'><div class='mv red'>{nd}</div><div class='ml'>Drift Events</div></div>
        <div class='mc card-y'><div class='mv yellow'>{nr}</div><div class='ml'>Retrains</div></div>
        <div class='mc card-g'><div class='mv green'>{nD}</div><div class='ml'>Deployed</div></div>
        <div class='mc card-b'><div class='mv blue'>{nR}</div><div class='ml'>Rejected</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>Full Decision History</div>", unsafe_allow_html=True)

    oc_map = {
        "NO_ACTION": "<span class='badge bg'>NO ACTION</span>",
        "DEPLOYED":  "<span class='badge bb'>DEPLOYED</span>",
        "REJECTED":  "<span class='badge by'>REJECTED</span>",
        "RETRAIN_FAILED": "<span class='badge br'>FAILED</span>",
        "—": "<span style='color:#64748b;font-size:.75rem;'>—</span>"
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
            <td>{oc_map.get(r['outcome'], r['outcome'])}</td>
            <td style='color:#94a3b8;font-size:.77rem;max-width:280px;'>{r['reason'][:75]}{'...' if len(r['reason'])>75 else ''}</td>
        </tr>"""
    tbl += "</tbody></table>"
    st.markdown(f"<div class='card'>{tbl}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='hr'><div class='sec'>Detailed View</div>", unsafe_allow_html=True)
    sel = st.selectbox("Pick month", [f"Month {m}" for m in range(1,7)], label_visibility="collapsed")
    dec = load_decision(int(sel.split()[-1]))
    if dec:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class='card card-b'>
                <div class='sec'>Decision Chain</div>
                <div style='font-size:.83rem;line-height:2.1;'>
                    <div>📍 <b>Drift:</b> {'Yes' if dec.get('drift_detected') else 'No'}</div>
                    <div>🔄 <b>Retrain:</b> {'Yes' if dec.get('retrain_decision') else 'No'}</div>
                    <div>🚀 <b>Deploy:</b> {str(dec.get('deploy_decision'))}</div>
                    <div>📊 <b>Model:</b> <span style='font-family:Space Mono,monospace;color:#00ff88;'>{dec.get('active_model','—')}</span></div>
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            oc = dec.get("outcome","—")
            oc_c = "#00ff88" if oc=="DEPLOYED" else "#ffbe00" if oc=="REJECTED" else "#3b82f6"
            st.markdown(f"""<div class='card card-{'g' if oc=='DEPLOYED' else 'y' if oc=='REJECTED' else 'b'}'>
                <div class='sec'>Final Outcome</div>
                <div style='font-family:Space Mono,monospace;font-size:1.5rem;color:{oc_c};margin:7px 0;'>{oc}</div>
                <div style='font-size:.77rem;color:#94a3b8;'>{dec.get("retrain_reason","—")}</div>
                {f'<div style="font-size:.75rem;color:#64748b;margin-top:5px;">{dec.get("deploy_reason","")}</div>' if dec.get("deploy_reason") else ""}
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 5 — DATA EXPLORER
# ═══════════════════════════════════════════════════════
elif page == "📁 Data Explorer":
    st.markdown("<div class='dg-title'>Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='dg-sub'>Explore monthly batch distributions and compare against baseline</div>", unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.warning("Dataset not found. Run `python scripts/prepare_real_data.py` first.")
    else:
        c1,c2 = st.columns(2)
        with c1:
            feat = st.selectbox("Feature", ["cibil_score","loan_amount","income_annum","loan_term",
                                             "residential_assets","luxury_assets","num_dependents"])
        with c2:
            cm   = st.selectbox("Compare Month", [3,4,5,6], format_func=lambda x: f"Month {x} vs Baseline")

        baseline = df[df["month"].isin([1,2])][feat]
        current  = df[df["month"]==cm][feat]

        fig_d = go.Figure()
        fig_d.add_trace(go.Histogram(x=baseline, name="Baseline (M1+M2)",
            marker_color="rgba(59,130,246,.6)", nbinsx=30, histnorm="probability density"))
        fig_d.add_trace(go.Histogram(x=current, name=f"Month {cm}",
            marker_color="rgba(255,59,92,.6)", nbinsx=30, histnorm="probability density"))
        fig_d.update_layout(**LAYOUT, barmode="overlay", height=300,
            title=dict(text=f"Distribution Comparison: {feat}", font=dict(size=13, color="#e8edf5")))
        st.plotly_chart(fig_d, use_container_width=True)

        st.markdown("<hr class='hr'><div class='sec'>Default Rate by Month</div>", unsafe_allow_html=True)
        md = df.groupby("month").agg(default_rate=("default_label","mean"), drift=("drift_type","first")).reset_index()
        fig_dr = go.Figure(go.Bar(
            x=[f"M{m}" for m in md["month"]], y=md["default_rate"],
            marker_color=["#ff3b5c" if d!="none" else "#3b82f6" for d in md["drift"]],
            text=[f"{r:.1%}" for r in md["default_rate"]],
            textposition="outside", textfont=dict(family="Space Mono", size=11)
        ))
        fig_dr.update_layout(**LAYOUT, height=250, yaxis=dict(**LAYOUT["yaxis"], tickformat=".0%"))
        st.plotly_chart(fig_dr, use_container_width=True)

        st.markdown("<div class='sec'>Sample Data</div>", unsafe_allow_html=True)
        sm = st.select_slider("Month", options=list(range(1,7)), format_func=lambda x: f"Month {x}", label_visibility="collapsed")
        st.dataframe(df[df["month"]==sm].head(12), use_container_width=True, hide_index=True)
