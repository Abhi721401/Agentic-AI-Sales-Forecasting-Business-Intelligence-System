"""pages/anomaly.py — Anomaly Detection"""

import streamlit as st
import pandas as pd
from modules.data_loader import load_and_clean_data
from modules.anomaly import (
    run_isolation_forest, run_autoencoder,
    combined_anomaly_summary, explain_anomaly,
)
from utils.charts import anomaly_scatter, LAYOUT
import plotly.express as px


def render():
    st.markdown('<div class="section-header">🚨 Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Two complementary methods are used:<br>
    <b>Isolation Forest</b> — ensemble of random trees; points isolated with fewer splits are anomalies.
    Fast, scalable, no distributional assumptions.<br>
    <b>Autoencoder</b> — neural network trained to reconstruct normal patterns;
    high reconstruction error signals anomaly. Captures non-linear structure.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    df, _ = load_and_clean_data()

    with st.sidebar:
        st.markdown("### ⚙️ Detection Settings")
        contamination = st.slider("Isolation Forest — contamination %", 1, 15, 5) / 100
        ae_pct        = st.slider("Autoencoder — threshold percentile", 85, 99, 95)
        stores        = sorted(df["Store"].unique())
        sel_store     = st.selectbox("Filter store (optional)", ["All"] + list(stores))

    fdf = df if sel_store == "All" else df[df["Store"] == int(sel_store)]

    tab1, tab2, tab3 = st.tabs(["🌲 Isolation Forest", "🧠 Autoencoder", "🔴 Combined View"])

    with tab1:
        with st.spinner("Running Isolation Forest..."):
            if_df = run_isolation_forest(fdf, contamination=contamination)

        n_anom = int(if_df["IF_Flag"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(if_df):,}")
        c2.metric("Anomalies Found", f"{n_anom:,}")
        c3.metric("Anomaly Rate", f"{n_anom/len(if_df)*100:.2f}%")

        st.plotly_chart(anomaly_scatter(if_df, flag_col="IF_Flag"), use_container_width=True)

        # Top anomalies with explanations
        st.markdown("#### 🔍 Top Anomalies with Explanations")
        top_if = if_df[if_df["IF_Flag"] == 1].nsmallest(20, "IF_Score")
        for _, row in top_if.head(10).iterrows():
            exp = explain_anomaly(row)
            sales = row.get("Weekly_Sales", 0)
            store = int(row.get("Store", 0))
            dept  = int(row.get("Dept", 0))
            date  = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
            with st.expander(f"Store {store} | Dept {dept} | {date} | ${sales:,.0f}"):
                st.markdown(exp)

        st.session_state["anomaly_df"] = if_df

    with tab2:
        with st.spinner("Training Autoencoder (NumPy implementation)..."):
            ae_df = run_autoencoder(fdf, threshold_percentile=ae_pct)

        n_ae = int(ae_df["AE_Flag"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(ae_df):,}")
        c2.metric("AE Anomalies", f"{n_ae:,}")
        c3.metric("Threshold Error", f"{ae_df['AE_Threshold'].iloc[0]:.4f}")

        # Reconstruction error distribution
        fig_err = px.histogram(
            ae_df, x="AE_Recon_Error", nbins=80,
            color_discrete_sequence=["#8b5cf6"],
            title="Autoencoder Reconstruction Error Distribution",
            marginal="box",
        )
        fig_err.add_vline(
            x=ae_df["AE_Threshold"].iloc[0],
            line_dash="dash", line_color="#ef4444",
            annotation_text=f"Threshold (p{ae_pct})",
        )
        fig_err.update_layout(**LAYOUT, height=400)
        st.plotly_chart(fig_err, use_container_width=True)

        st.plotly_chart(anomaly_scatter(ae_df, flag_col="AE_Flag"), use_container_width=True)

    with tab3:
        with st.spinner("Running both detectors..."):
            if_df2 = run_isolation_forest(fdf, contamination=contamination)
            ae_df2 = run_autoencoder(fdf, threshold_percentile=ae_pct)
            combo  = combined_anomaly_summary(if_df2, ae_df2)

        n_both = int(combo["Both_Flagged"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("IF Anomalies",      str(int(combo["IF_Flag"].sum())))
        c2.metric("AE Anomalies",      str(int(combo["AE_Flag"].sum())))
        c3.metric("Both Flagged 🔴",   str(n_both))

        st.markdown("""
        Anomalies flagged by **both** methods are classified as 🔴 High Confidence.
        Single-method flags are 🟡 and may warrant secondary investigation.
        """)

        # Confidence breakdown
        fig_conf = px.pie(
            combo["Confidence"].value_counts().reset_index(),
            names="Confidence", values="count",
            color_discrete_sequence=["#ef4444", "#f59e0b"],
            title="Anomaly Confidence Classification",
        )
        fig_conf.update_layout(**LAYOUT, height=380)
        st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown("#### Combined Anomaly Table")
        st.dataframe(combo.head(50), use_container_width=True)
