"""
RetailGPT: Agentic AI Sales Forecasting and Business Intelligence System
Main Streamlit Application Entry Point
"""

import streamlit as st
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RetailGPT — Agentic AI Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
    section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.85rem; }

    /* KPI card */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
    .kpi-title { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #f1f5f9; font-size: 2rem; font-weight: 700; margin: 6px 0; }
    .kpi-delta { font-size: 0.82rem; }
    .kpi-delta.pos { color: #22c55e; }
    .kpi-delta.neg { color: #ef4444; }

    /* Section headers */
    .section-header {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
        border-left: 4px solid #3b82f6;
        padding-left: 12px;
        margin-bottom: 18px;
    }

    /* Info box */
    .info-box {
        background: #1e293b;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        color: #93c5fd;
        font-size: 0.9rem;
    }

    /* Plotly chart container */
    .js-plotly-plot { border-radius: 10px; }

    /* Metric override */
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:2.5rem;'>🛒</div>
        <div style='color:#f1f5f9; font-size:1.2rem; font-weight:700;'>RetailGPT</div>
        <div style='color:#64748b; font-size:0.75rem;'>Agentic AI Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Home Dashboard",
            "🔍  Data Exploration",
            "📈  Regression Analysis",
            "📅  Forecasting",
            "🚨  Anomaly Detection",
            "🧠  AI Insights (XAI)",
            "🤖  Agentic AI Assistant",
            "📄  Report Generation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569; font-size:0.72rem; text-align:center;'>
        Walmart Sales Dataset<br>45 Stores · 81 Depts · 2010–2012
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ───────────────────────────────────────────────────────────────
if "🏠" in page:
    from pages.home import render
    render()
elif "🔍" in page:
    from pages.data_exploration import render
    render()
elif "📈" in page:
    from pages.regression import render
    render()
elif "📅" in page:
    from pages.forecasting import render
    render()
elif "🚨" in page:
    from pages.anomaly import render
    render()
elif "🧠" in page:
    from pages.xai import render
    render()
elif "🤖" in page:
    from pages.agent import render
    render()
elif "📄" in page:
    from pages.report import render
    render()
