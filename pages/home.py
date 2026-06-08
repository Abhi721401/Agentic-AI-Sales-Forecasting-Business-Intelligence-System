"""pages/home.py — Home Dashboard (complete)"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.data_loader import load_and_clean_data
from utils.charts import sales_trend, store_comparison_bar, LAYOUT
from utils.helpers import fmt_currency, info_card


def kpi(col, title, value, delta=None, delta_pos=True):
    delta_html = ""
    if delta:
        cls   = "pos" if delta_pos else "neg"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    col.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>""",
        unsafe_allow_html=True,
    )


def render():
    st.markdown('<div class="section-header">🏠 Home Dashboard</div>', unsafe_allow_html=True)

    with st.spinner("Loading & cleaning data..."):
        df, log = load_and_clean_data()

    # ── KPI Strip ──────────────────────────────────────────────────────────────
    total   = df["Weekly_Sales"].sum()
    avg_wk  = df["Weekly_Sales"].mean()
    peak_wk = df["Weekly_Sales"].max()
    n_stores = df["Store"].nunique()
    n_depts  = df["Dept"].nunique()
    h_avg    = df[df["IsHoliday"] == 1]["Weekly_Sales"].mean()
    nh_avg   = df[df["IsHoliday"] == 0]["Weekly_Sales"].mean()
    holiday_uplift = (h_avg / nh_avg - 1) * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, "Total Revenue",    fmt_currency(total, compact=True))
    kpi(c2, "Avg Weekly Sales", fmt_currency(avg_wk, compact=True))
    kpi(c3, "Peak Week Sales",  fmt_currency(peak_wk, compact=True))
    kpi(c4, "Stores",           str(n_stores))
    kpi(c5, "Departments",      str(n_depts))
    kpi(c6, "Holiday Uplift",   f"{holiday_uplift:.1f}%", "vs non-holiday", True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Data status banner ─────────────────────────────────────────────────────
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    real_data = os.path.exists(os.path.join(data_dir, "train.csv")) and \
                os.path.getsize(os.path.join(data_dir, "train.csv")) > 5_000_000  # >5MB = real

    if real_data:
        info_card("📂 Kaggle Data Loaded", f"{len(df):,} records from the official Walmart competition dataset.", "#22c55e")
    else:
        info_card("🧪 Synthetic Demo Data",
                  "Real Kaggle CSVs not found. Showing synthetic data. "
                  "Download from kaggle.com/c/walmart-recruiting-store-sales-forecasting/data → place in data/.",
                  "#f59e0b")

    # ── Main trend chart ───────────────────────────────────────────────────────
    st.plotly_chart(
        sales_trend(df, title="📈 Total Weekly Sales — All 45 Stores (2010–2012)"),
        use_container_width=True,
    )

    # ── Row 2: Store bar + Monthly by year ────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.plotly_chart(store_comparison_bar(df, top_n=15), use_container_width=True)

    with col_r:
        monthly = (
            df.groupby(["Year", "Month"])["Weekly_Sales"]
            .sum()
            .reset_index()
        )
        monthly["Period"] = (
            monthly["Year"].astype(str) + "-"
            + monthly["Month"].astype(str).str.zfill(2)
        )
        monthly["Year"] = monthly["Year"].astype(str)
        fig_m = px.bar(
            monthly, x="Period", y="Weekly_Sales",
            color="Year", barmode="group",
            color_discrete_sequence=["#3b82f6", "#22c55e", "#f59e0b"],
            labels={"Weekly_Sales": "Total Sales ($)"},
        )
        fig_m.update_layout(**LAYOUT, title="Monthly Revenue by Year", height=420)
        fig_m.update_xaxes(tickangle=45)
        st.plotly_chart(fig_m, use_container_width=True)

    # ── Row 3: Store type pie + Holiday box ───────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        type_sales = df.groupby("Type")["Weekly_Sales"].sum().reset_index()
        fig_pie = px.pie(
            type_sales, names="Type", values="Weekly_Sales",
            color_discrete_sequence=["#3b82f6", "#22c55e", "#f59e0b"],
            hole=0.4,
        )
        fig_pie.update_layout(**LAYOUT, title="Revenue by Store Type", height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        df_box = df[["IsHoliday", "Weekly_Sales"]].copy()
        df_box["Week Type"] = df_box["IsHoliday"].map({1: "Holiday", 0: "Non-Holiday"})
        fig_box = px.box(
            df_box, x="Week Type", y="Weekly_Sales",
            color="Week Type",
            color_discrete_map={"Holiday": "#f59e0b", "Non-Holiday": "#3b82f6"},
            points="outliers",
        )
        fig_box.update_layout(**LAYOUT, title="Sales: Holiday vs Non-Holiday", height=380)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Row 4: Dept top-10 + Store size vs sales ──────────────────────────────
    col_p, col_q = st.columns(2)

    with col_p:
        top_depts = (
            df.groupby("Dept")["Weekly_Sales"].sum()
            .nlargest(10).reset_index()
        )
        fig_d = px.bar(
            top_depts, x="Dept", y="Weekly_Sales",
            color="Weekly_Sales", color_continuous_scale="Blues",
            labels={"Weekly_Sales": "Total Sales ($)"},
        )
        fig_d.update_coloraxes(showscale=False)
        fig_d.update_layout(**LAYOUT, title="Top 10 Departments by Revenue", height=380)
        st.plotly_chart(fig_d, use_container_width=True)

    with col_q:
        store_agg = (
            df.groupby(["Store", "Type", "Size"])["Weekly_Sales"]
            .sum().reset_index()
        )
        fig_sc = px.scatter(
            store_agg, x="Size", y="Weekly_Sales",
            color="Type", size="Weekly_Sales",
            color_discrete_sequence=["#3b82f6", "#22c55e", "#f59e0b"],
            hover_data=["Store"],
            labels={"Weekly_Sales": "Total Revenue ($)", "Size": "Store Size (sq ft)"},
        )
        fig_sc.update_layout(**LAYOUT, title="Store Size vs Total Revenue", height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Quarterly trend ────────────────────────────────────────────────────────
    df["YQ"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)
    quarterly = df.groupby("YQ")["Weekly_Sales"].sum().reset_index()
    fig_q = px.line(
        quarterly, x="YQ", y="Weekly_Sales",
        markers=True, color_discrete_sequence=["#8b5cf6"],
        labels={"Weekly_Sales": "Total Sales ($)", "YQ": "Quarter"},
    )
    fig_q.update_layout(**LAYOUT, title="📊 Quarterly Revenue Trend", height=360)
    st.plotly_chart(fig_q, use_container_width=True)

    # ── Named holiday analysis ─────────────────────────────────────────────────
    st.markdown("### 🎄 Named Holiday Sales Impact")
    holiday_cols = [c for c in df.columns if c.startswith("Is_")]
    if holiday_cols:
        h_data = []
        for col in holiday_cols:
            name = col.replace("Is_", "").replace("_", " ")
            avg_h  = df[df[col] == 1]["Weekly_Sales"].mean()
            avg_nh = df[df[col] == 0]["Weekly_Sales"].mean()
            uplift = (avg_h / avg_nh - 1) * 100 if avg_nh > 0 else 0
            h_data.append({"Holiday": name, "Avg Sales (Holiday)": avg_h,
                           "Avg Sales (Non-Holiday)": avg_nh, "Uplift %": uplift})
        h_df = pd.DataFrame(h_data)
        fig_h = px.bar(
            h_df, x="Holiday", y="Uplift %",
            color="Uplift %", color_continuous_scale="Greens",
            labels={"Uplift %": "Sales Uplift (%)"},
        )
        fig_h.update_coloraxes(showscale=False)
        fig_h.update_layout(**LAYOUT, title="Sales Uplift by Named Holiday", height=360)
        st.plotly_chart(fig_h, use_container_width=True)

    # ── Cleaning log ───────────────────────────────────────────────────────────
    with st.expander("📋 Data Cleaning & Feature Engineering Log", expanded=False):
        for entry in log:
            st.markdown(f"- {entry}")
        st.caption(f"Dataset shape after cleaning: {df.shape[0]:,} rows × {df.shape[1]} columns")
