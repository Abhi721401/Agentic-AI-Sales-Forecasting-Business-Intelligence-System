"""pages/forecasting.py — Complete ARIMA & SARIMA Forecasting Page"""

import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import load_and_clean_data, get_store_series, get_store_dept_series
from modules.forecasting import (
    fit_arima, fit_sarima, compare_models, adf_test,
    compute_acf_pacf, HORIZONS,
)
from utils.charts import forecast_plot, acf_pacf_plot, LAYOUT
from utils.helpers import info_card, fmt_currency
import plotly.express as px
import plotly.graph_objects as go


def render():
    st.markdown('<div class="section-header">📅 Forecasting — ARIMA & SARIMA</div>', unsafe_allow_html=True)
    info_card(
        "Forecasting Methodology",
        "Stationarity is checked via the Augmented Dickey-Fuller test (H₀: unit root). "
        "ARIMA order (p,d,q) is selected by AIC grid search. "
        "SARIMA adds seasonal terms (P,D,Q)ₛ with m=52 for annual weekly seasonality. "
        "95% prediction intervals are derived from the ARIMA state-space representation.",
        "#3b82f6",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    df, _ = load_and_clean_data()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Forecast Settings")
        stores = sorted(df["Store"].unique())
        sel_store = st.selectbox("Store", stores, index=0)

        mode = st.radio("Granularity", ["Store-level (aggregated)", "Store + Department"])
        if mode == "Store + Department":
            depts = sorted(df[df["Store"] == sel_store]["Dept"].unique())
            sel_dept = st.selectbox("Department", depts)
        else:
            sel_dept = None

        horizon_label = st.selectbox("Forecast Horizon", list(HORIZONS.keys()), index=1)
        run_sarima = st.checkbox("Include SARIMA", value=True)
        st.markdown("---")
        st.markdown(
            "**💡 Tip:** SARIMA takes longer (~30s) due to seasonal grid search. "
            "Uncheck for quick ARIMA-only forecasts."
        )

    # ── Load series ────────────────────────────────────────────────────────────
    if sel_dept:
        ts = get_store_dept_series(df, sel_store, sel_dept)
        series_label = f"Store {sel_store} — Dept {sel_dept}"
    else:
        ts = get_store_series(df, sel_store)
        series_label = f"Store {sel_store} (all depts aggregated)"

    if ts.empty or len(ts) < 12:
        st.error("Not enough data for this selection (need ≥ 12 weeks). Choose a different store/dept.")
        return

    st.info(f"📊 **{series_label}** — {len(ts)} weekly observations | "
            f"{ts.index.min().date()} → {ts.index.max().date()} | "
            f"Total Revenue: {fmt_currency(ts.sum(), compact=True)}")

    # ── Historical chart ───────────────────────────────────────────────────────
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode="lines", name="Historical Sales",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
    ))
    fig_hist.update_layout(**LAYOUT, title=f"Historical Weekly Sales — {series_label}", height=360)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Stationarity ───────────────────────────────────────────────────────────
    st.markdown("### 📐 Stationarity Analysis (ADF Test)")
    adf = adf_test(ts)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ADF Statistic",  adf["ADF Statistic"])
    col2.metric("p-value",        adf["p-value"])
    col3.metric("Critical (5%)",  adf["Critical Values"].get("5%", "N/A"))
    col4.metric("Verdict", "✅ Stationary" if adf["Stationary"] else "❌ Non-stationary → d=1")

    if not adf["Stationary"]:
        info_card(
            "Differencing Required",
            "ADF p-value > 0.05 → fail to reject H₀ (unit root present). "
            "First-order differencing (d=1) will be applied. "
            "If differenced series is still non-stationary, d=2 is tried.",
            "#f59e0b",
        )
    else:
        info_card("Series is Stationary", "ADF p < 0.05 → reject H₀. d=0 (no differencing needed).", "#22c55e")

    # ── ACF / PACF ─────────────────────────────────────────────────────────────
    st.markdown("### 🔁 ACF & PACF Analysis")
    acf_vals, pacf_vals = compute_acf_pacf(ts, nlags=40)
    st.plotly_chart(acf_pacf_plot(acf_vals, pacf_vals), use_container_width=True)
    st.caption(
        "**Reading the plots:** "
        "PACF cuts off at lag p → AR(p) structure (feeds ARIMA p parameter). "
        "ACF cuts off at lag q → MA(q) structure (feeds ARIMA q parameter). "
        "Dashed red lines = 95% Bartlett confidence bounds (±1.96/√n)."
    )

    # ── Differenced series (if needed) ────────────────────────────────────────
    if not adf["Stationary"]:
        ts_diff = ts.diff().dropna()
        adf_diff = adf_test(ts_diff)
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Scatter(x=ts_diff.index, y=ts_diff.values,
                                      mode="lines", name="Differenced",
                                      line=dict(color="#22c55e", width=1.5)))
        fig_diff.update_layout(**LAYOUT, title="First-Order Differenced Series", height=300)
        with st.expander("📉 View differenced series"):
            st.plotly_chart(fig_diff, use_container_width=True)
            st.caption(f"ADF on differenced: p = {adf_diff['p-value']} — "
                       f"{'Stationary ✅' if adf_diff['Stationary'] else 'Still non-stationary (d=2 will be used)'}")

    # ── Fit models ─────────────────────────────────────────────────────────────
    st.markdown("### 🔮 Model Fitting & Forecast")

    if run_sarima:
        col_a, col_s = st.columns(2)
    else:
        col_a = st.container()

    arima_res = None
    sarima_res = None

    with col_a:
        st.markdown("#### ARIMA")
        with st.spinner(f"Fitting ARIMA for {series_label}..."):
            try:
                arima_res = fit_arima(ts, horizon_label)
                m = arima_res["metrics"]
                st.success(f"Best order: **ARIMA{arima_res['order']}** | AIC: {arima_res['aic']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", fmt_currency(m.get("RMSE", 0)))
                c2.metric("MAE",  fmt_currency(m.get("MAE", 0)))
                c3.metric("MAPE", f"{m.get('MAPE (%)', 0):.2f}%")
                st.plotly_chart(
                    forecast_plot(arima_res["historical"], arima_res["forecast"], "ARIMA"),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"ARIMA fitting failed: {e}")

    if run_sarima:
        with col_s:
            st.markdown("#### SARIMA (m=52)")
            with st.spinner(f"Fitting SARIMA (this takes ~30s)..."):
                try:
                    sarima_res = fit_sarima(ts, horizon_label, m=52)
                    m = sarima_res["metrics"]
                    st.success(
                        f"Best: **SARIMA{sarima_res['order']}×{sarima_res['seasonal_order']}** | "
                        f"AIC: {sarima_res['aic']}"
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("RMSE", fmt_currency(m.get("RMSE", 0)))
                    c2.metric("MAE",  fmt_currency(m.get("MAE", 0)))
                    c3.metric("MAPE", f"{m.get('MAPE (%)', 0):.2f}%")
                    st.plotly_chart(
                        forecast_plot(sarima_res["historical"], sarima_res["forecast"], "SARIMA"),
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"SARIMA fitting failed: {e}")

    # ── Comparison ─────────────────────────────────────────────────────────────
    if arima_res and sarima_res:
        st.markdown("### ⚖️ Model Comparison")
        comp = compare_models(arima_res, sarima_res)
        st.dataframe(comp.set_index("Model"), use_container_width=True)

        winner = comp.loc[comp["AIC"].idxmin(), "Model"]
        info_card(
            "Winner by AIC",
            f"{winner} achieves lower AIC, indicating better fit penalised for complexity. "
            "SARIMA generally wins on retail data because it captures annual holiday cycles. "
            "Use ARIMA when speed matters or the series is short (<52 observations).",
            "#22c55e",
        )

    # ── Forecast table ─────────────────────────────────────────────────────────
    if arima_res:
        st.markdown("### 📋 ARIMA Forecast Values (First 12 weeks)")
        fc = arima_res["forecast"].head(12).copy()
        fc["Forecast"]  = fc["Forecast"].map("${:,.0f}".format)
        fc["Lower_CI"]  = fc["Lower_CI"].map("${:,.0f}".format)
        fc["Upper_CI"]  = fc["Upper_CI"].map("${:,.0f}".format)
        st.dataframe(fc, use_container_width=True, hide_index=True)

    # ── Multi-store comparison ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏪 Multi-Store Sales Comparison")
    compare_stores = st.multiselect(
        "Compare stores on same chart",
        sorted(df["Store"].unique()),
        default=sorted(df["Store"].unique())[:3],
    )
    if compare_stores:
        fig_multi = go.Figure()
        for s in compare_stores:
            s_ts = get_store_series(df, s)
            fig_multi.add_trace(go.Scatter(
                x=s_ts.index, y=s_ts.values,
                mode="lines", name=f"Store {s}",
            ))
        fig_multi.update_layout(**LAYOUT, title="Store Sales Comparison", height=420)
        st.plotly_chart(fig_multi, use_container_width=True)

    # ── Save to session ────────────────────────────────────────────────────────
    if arima_res:
        st.session_state["forecast_arima"] = arima_res
    if sarima_res:
        st.session_state["forecast_sarima"] = sarima_res
