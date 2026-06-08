"""pages/data_exploration.py — Interactive EDA"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules.data_loader import load_and_clean_data, outlier_summary
from utils.charts import sales_trend, dept_heatmap, LAYOUT


def render():
    st.markdown('<div class="section-header">🔍 Data Exploration</div>', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        df, _ = load_and_clean_data()

    # ── Sidebar filters ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔧 Filters")
        stores = sorted(df["Store"].unique())
        sel_stores = st.multiselect("Stores", stores, default=stores[:5])

        depts = sorted(df["Dept"].unique())
        sel_depts = st.multiselect("Departments", depts, default=depts[:5])

        year_range = st.slider("Year", int(df["Year"].min()), int(df["Year"].max()),
                               (int(df["Year"].min()), int(df["Year"].max())))

    fdf = df[
        df["Store"].isin(sel_stores) &
        df["Dept"].isin(sel_depts) &
        df["Year"].between(*year_range)
    ]

    st.caption(f"Showing {len(fdf):,} rows filtered from {len(df):,} total")

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "📈 Trends", "🌡️ Correlations", "⚠️ Outliers"])

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            fig_hist = px.histogram(
                fdf, x="Weekly_Sales", nbins=60,
                color_discrete_sequence=["#3b82f6"],
                title="Weekly Sales Distribution",
                marginal="box",
            )
            fig_hist.update_layout(**LAYOUT, height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_r:
            fig_log = px.histogram(
                fdf[fdf["Weekly_Sales"] > 0],
                x=np.log1p(fdf[fdf["Weekly_Sales"] > 0]["Weekly_Sales"]),
                nbins=60,
                color_discrete_sequence=["#22c55e"],
                title="Log(Weekly Sales) Distribution",
                marginal="box",
            )
            fig_log.update_layout(**LAYOUT, height=400)
            fig_log.update_xaxes(title="log(1 + Weekly Sales)")
            st.plotly_chart(fig_log, use_container_width=True)

        # Missing values heatmap
        st.markdown("**Missing Values After Cleaning**")
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing = missing[missing["Missing"] > 0]
        if missing.empty:
            st.success("✅ No missing values remain after cleaning.")
        else:
            st.dataframe(missing, use_container_width=True)

        # Descriptive statistics
        st.markdown("**Descriptive Statistics**")
        st.dataframe(
            fdf[["Weekly_Sales", "Temperature", "Fuel_Price", "CPI",
                 "Unemployment", "Total_MarkDown"]].describe().round(2),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            sales_trend(fdf, title="Filtered Sales Trend"),
            use_container_width=True,
        )

        # Per-store trends
        store_weekly = (
            fdf.groupby(["Date", "Store"])["Weekly_Sales"]
            .sum().reset_index()
        )
        fig_multi = px.line(
            store_weekly, x="Date", y="Weekly_Sales",
            color="Store", color_discrete_sequence=px.colors.qualitative.Set2,
            title="Sales by Store Over Time",
        )
        fig_multi.update_layout(**LAYOUT, height=450)
        st.plotly_chart(fig_multi, use_container_width=True)

        # Department heatmap — only first selected store
        if sel_stores:
            st.markdown(f"**Dept × Month Heatmap — Store {sel_stores[0]}**")
            st.plotly_chart(dept_heatmap(fdf, store=sel_stores[0]), use_container_width=True)

    with tab3:
        num_cols = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI",
                    "Unemployment", "Total_MarkDown", "Size"]
        num_cols = [c for c in num_cols if c in fdf.columns]
        corr = fdf[num_cols].corr()

        fig_corr = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
        )
        fig_corr.update_layout(**LAYOUT, height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter: MarkDown vs Sales
        col_l, col_r = st.columns(2)
        with col_l:
            fig_sc = px.scatter(
                fdf.sample(min(2000, len(fdf)), random_state=42),
                x="Total_MarkDown", y="Weekly_Sales",
                color="IsHoliday",
                color_continuous_scale=["#3b82f6", "#f59e0b"],
                title="Total MarkDown vs Weekly Sales",
                opacity=0.5,
            )
            fig_sc.update_layout(**LAYOUT, height=380)
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_r:
            fig_sc2 = px.scatter(
                fdf.sample(min(2000, len(fdf)), random_state=99),
                x="Unemployment", y="Weekly_Sales",
                color="Type",
                color_discrete_sequence=["#3b82f6", "#22c55e", "#f59e0b"],
                title="Unemployment vs Weekly Sales",
                opacity=0.5,
            )
            fig_sc2.update_layout(**LAYOUT, height=380)
            st.plotly_chart(fig_sc2, use_container_width=True)

    with tab4:
        st.markdown("""
        <div class="info-box">
        Outliers are identified using IQR (1.5× rule) and Z-score (>3σ) methods.
        They are <strong>not removed</strong> — most correspond to legitimate holiday demand spikes
        (Thanksgiving, Christmas) which are genuine sales signal, not noise.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        out_df = outlier_summary(df)
        n_out  = len(out_df)
        n_holiday = (out_df["Likely_Cause"].str.contains("Bowl|Day|Thanksgiving|Christmas")).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Outliers (IQR)", f"{n_out:,}")
        c2.metric("Holiday-Driven", f"{n_holiday:,}", f"{n_holiday/n_out*100:.0f}%")
        c3.metric("Promotion-Driven", str(len(out_df[out_df["Likely_Cause"] == "Promotional Markdown"])))

        cause_counts = out_df["Likely_Cause"].value_counts().reset_index()
        cause_counts.columns = ["Cause", "Count"]
        fig_out = px.bar(cause_counts, x="Cause", y="Count",
                         color="Count", color_continuous_scale="Reds",
                         title="Outlier Root Cause Classification")
        fig_out.update_layout(**LAYOUT, height=380)
        st.plotly_chart(fig_out, use_container_width=True)

        st.dataframe(out_df.head(50), use_container_width=True)
