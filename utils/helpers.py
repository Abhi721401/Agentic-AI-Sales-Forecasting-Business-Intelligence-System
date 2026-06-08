"""
utils/helpers.py
────────────────
Shared utility functions used across all pages:
  - Metric formatting
  - Session state management helpers
  - Color-coded metric delta rendering
  - Pandas display helpers
  - Groq API key loader (env + sidebar)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ── API key ────────────────────────────────────────────────────────────────────

def get_groq_api_key() -> str:
    """
    Priority: st.session_state > environment variable > empty string.
    The sidebar in agent.py sets st.session_state["groq_api_key"].
    """
    if "groq_api_key" in st.session_state and st.session_state["groq_api_key"]:
        return st.session_state["groq_api_key"]
    return os.environ.get("GROQ_API_KEY", "")


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt_currency(val: float, compact: bool = False) -> str:
    if compact:
        if abs(val) >= 1e9:
            return f"${val/1e9:.2f}B"
        if abs(val) >= 1e6:
            return f"${val/1e6:.1f}M"
        if abs(val) >= 1e3:
            return f"${val/1e3:.1f}K"
    return f"${val:,.2f}"


def fmt_pct(val: float, decimals: int = 1) -> str:
    return f"{val:.{decimals}f}%"


def fmt_number(val: float, decimals: int = 0) -> str:
    return f"{val:,.{decimals}f}"


# ── Session state ──────────────────────────────────────────────────────────────

def init_session_state(defaults: dict):
    """Initialise session state keys without overwriting existing values."""
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_model_cache():
    """Clear cached model results from session state."""
    keys_to_clear = [
        "regression_results", "forecast_arima", "forecast_sarima",
        "anomaly_df", "shap_vals", "X_sample", "imp_df",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)


# ── Pandas display helpers ─────────────────────────────────────────────────────

def highlight_positive_negative(df: pd.DataFrame, col: str) -> pd.DataFrame.style:
    """Return a Styler that colours positive values green and negative red."""
    def color(val):
        try:
            c = "#22c55e" if float(val) > 0 else "#ef4444"
            return f"color: {c}"
        except (TypeError, ValueError):
            return ""
    return df.style.applymap(color, subset=[col])


def top_n_table(df: pd.DataFrame, group_col: str, value_col: str,
                n: int = 10, agg: str = "sum") -> pd.DataFrame:
    """Return top-N rows grouped by group_col, sorted by value_col."""
    grouped = df.groupby(group_col)[value_col].agg(agg).nlargest(n).reset_index()
    grouped.columns = [group_col, f"{agg.title()} {value_col}"]
    return grouped


# ── Holiday calendar helper ────────────────────────────────────────────────────

HOLIDAY_MAP = {
    pd.Timestamp("2010-02-12"): "Super Bowl",
    pd.Timestamp("2011-02-11"): "Super Bowl",
    pd.Timestamp("2012-02-10"): "Super Bowl",
    pd.Timestamp("2010-09-10"): "Labor Day",
    pd.Timestamp("2011-09-09"): "Labor Day",
    pd.Timestamp("2012-09-07"): "Labor Day",
    pd.Timestamp("2010-11-26"): "Thanksgiving",
    pd.Timestamp("2011-11-25"): "Thanksgiving",
    pd.Timestamp("2012-11-23"): "Thanksgiving",
    pd.Timestamp("2010-12-31"): "Christmas",
    pd.Timestamp("2011-12-30"): "Christmas",
    pd.Timestamp("2012-12-28"): "Christmas",
}


def get_holiday_name(date: pd.Timestamp) -> str:
    return HOLIDAY_MAP.get(date, "")


# ── Model performance colour coding ───────────────────────────────────────────

def r2_badge(r2: float) -> str:
    """Return a colour-coded badge string for R² values."""
    if r2 >= 0.85:
        return f"🟢 {r2:.4f} (Excellent)"
    elif r2 >= 0.70:
        return f"🟡 {r2:.4f} (Good)"
    elif r2 >= 0.50:
        return f"🟠 {r2:.4f} (Moderate)"
    else:
        return f"🔴 {r2:.4f} (Weak)"


def mape_badge(mape: float) -> str:
    if mape <= 5:
        return f"🟢 {mape:.2f}% (Excellent)"
    elif mape <= 10:
        return f"🟡 {mape:.2f}% (Good)"
    elif mape <= 20:
        return f"🟠 {mape:.2f}% (Acceptable)"
    else:
        return f"🔴 {mape:.2f}% (Poor)"


# ── Streamlit info cards ───────────────────────────────────────────────────────

def info_card(title: str, content: str, color: str = "#3b82f6"):
    st.markdown(
        f"""
        <div style="border-left: 4px solid {color}; background: #1e293b;
                    padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 12px;">
            <div style="color: {color}; font-weight: 700; font-size: 0.9rem;">{title}</div>
            <div style="color: #e2e8f0; font-size: 0.85rem; margin-top: 4px;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def success_card(content: str):
    info_card("✅ Success", content, "#22c55e")


def warning_card(content: str):
    info_card("⚠️ Warning", content, "#f59e0b")


def error_card(content: str):
    info_card("❌ Error", content, "#ef4444")
