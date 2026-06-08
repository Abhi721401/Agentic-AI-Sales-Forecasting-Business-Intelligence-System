"""pages/report.py — Complete PDF Report Generation Page"""

import streamlit as st
import pandas as pd
from datetime import datetime
from modules.data_loader import load_and_clean_data
from modules.report_generator import generate_report, FPDF_AVAILABLE
from utils.helpers import info_card, fmt_currency


def render():
    st.markdown('<div class="section-header">📄 Automated Report Generation</div>', unsafe_allow_html=True)
    info_card(
        "Report Contents",
        "Generates a professional multi-page PDF report. "
        "Run Forecasting and Anomaly Detection pages first to enrich the report with those results. "
        "All sections are optional and can be toggled below.",
        "#3b82f6",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if not FPDF_AVAILABLE:
        st.error("fpdf2 not installed.")
        st.code("pip install fpdf2")
        return

    df, log = load_and_clean_data()

    # ── Config columns ─────────────────────────────────────────────────────────
    col_cfg, col_preview = st.columns([3, 2])

    with col_cfg:
        st.markdown("### ⚙️ Report Configuration")

        report_title = st.text_input("Report Title", "RetailGPT Analytics Report")
        analyst_name = st.text_input("Analyst Name", "Data Science Team")

        st.markdown("**Sections to include:**")
        inc_exec    = st.checkbox("Executive Summary KPIs", value=True)
        inc_data    = st.checkbox("Dataset Overview & Holiday Uplift", value=True)
        inc_reg     = st.checkbox("Regression Model Performance", value=True)
        inc_fc      = st.checkbox("Forecasting Results", value=True)
        inc_anom    = st.checkbox("Anomaly Detection", value=True)
        inc_recs    = st.checkbox("AI Recommendations", value=True)
        inc_method  = st.checkbox("Methodology Notes", value=True)

        st.markdown("**Custom Recommendations (optional):**")
        custom_recs = st.text_area(
            "Leave blank for auto-generated",
            height=120,
            placeholder=(
                "1. Increase Thanksgiving inventory by 25% for Stores 4, 20, 28 based on SARIMA forecast.\n"
                "2. MarkDown3 shows lowest ROI — reallocate to MarkDown1 and MarkDown5.\n"
                "3. ..."
            ),
        )

    with col_preview:
        st.markdown("### 📊 Snapshot KPIs")
        total    = df["Weekly_Sales"].sum()
        avg_wk   = df["Weekly_Sales"].mean()
        top_s    = int(df.groupby("Store")["Weekly_Sales"].sum().idxmax())
        top_d    = int(df.groupby("Dept")["Weekly_Sales"].sum().idxmax())
        h_uplift = (
            df[df["IsHoliday"] == 1]["Weekly_Sales"].mean() /
            df[df["IsHoliday"] == 0]["Weekly_Sales"].mean() - 1
        ) * 100

        st.metric("Total Revenue",    fmt_currency(total, compact=True))
        st.metric("Avg Weekly Sales", fmt_currency(avg_wk))
        st.metric("Top Store",        f"Store {top_s}")
        st.metric("Top Dept",         f"Dept {top_d}")
        st.metric("Holiday Uplift",   f"+{h_uplift:.1f}%")
        st.metric("Date Range",
                  f"{df['Date'].min().date()} → {df['Date'].max().date()}")

        # Session state status
        # st.markdown("**Cached Results:**")
        # fc_ok   = "✅" if st.session_state.get("forecast_arima") else "⚠️ (run Forecasting page)"
        # anom_ok = "✅" if st.session_state.get("anomaly_df")     else "⚠️ (run Anomaly page)"
        # st.markdown(f"- Forecast: {fc_ok}")
        # st.markdown(f"- Anomalies: {anom_ok}")
        # Session state status
        st.markdown("**Cached Results:**")

        forecast_arima = st.session_state.get("forecast_arima")
        anomaly_df = st.session_state.get("anomaly_df")

        fc_ok = "✅" if forecast_arima is not None else "⚠️ (run Forecasting page)"

        anom_ok = ("✅"
                   if anomaly_df is not None and not anomaly_df.empty
                   else "⚠️ (run Anomaly page)"
                   )

        st.markdown(f"- Forecast: {fc_ok}")
        st.markdown(f"- Anomalies: {anom_ok}")

    st.markdown("---")

    # ── Generate button ────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        generate = st.button("🖨️ Generate PDF Report", type="primary", use_container_width=True)

    with col_info:
        st.caption(
            "PDF is generated server-side using fpdf2. "
            "Typical generation time: 5-15 seconds depending on included sections."
        )

    if generate:
        # Gather cached analysis results
        reg_results = None
        fc_arima    = st.session_state.get("forecast_arima")
        fc_sarima   = st.session_state.get("forecast_sarima")
        anomaly_df  = st.session_state.get("anomaly_df")

        progress = st.progress(0, "Starting...")

        if inc_reg:
            progress.progress(10, "Training regression models...")
            try:
                from modules.regression import train_all_models
                reg_results = train_all_models(df)
            except Exception as e:
                st.warning(f"Regression section skipped: {e}")

        progress.progress(40, "Assembling report sections...")

        try:
            progress.progress(60, "Rendering PDF...")
            pdf_bytes = generate_report(
                df=df,
                regression_results=reg_results  if inc_reg  else None,
                forecast_arima=fc_arima         if inc_fc   else None,
                forecast_sarima=fc_sarima       if inc_fc   else None,
                anomaly_df=anomaly_df           if inc_anom else None,
                recommendations=custom_recs     if inc_recs else "",
            )
            progress.progress(100, "Complete!")

            st.success(f"✅ Report ready — {len(pdf_bytes)/1024:.0f} KB")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"retailgpt_report_{timestamp}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            progress.empty()
            st.error(f"Report generation failed: {e}")
            st.exception(e)

    # ── Report structure reference ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Full Report Structure")
    structure = [
        ["Cover Page",                  "Title, analyst name, generation date, platform logo"],
        ["Executive Summary",           "5 KPI cards: revenue, avg sales, top store/dept, record count"],
        ["Dataset Overview",            "Store/dept counts, date range, holiday uplift calculation"],
        ["Regression Performance",      "Metrics table (R², RMSE, MAE, MAPE) for OLS/Ridge/Lasso"],
        ["Coefficient Interpretation",  "Top drivers table; Ridge vs Lasso comparison note"],
        ["Forecast — ARIMA",            "Model order, AIC, metrics, 6-week forecast table with 95% CI"],
        ["Forecast — SARIMA",           "Seasonal order, comparison with ARIMA, recommendation"],
        ["Anomaly Detection",           "Anomaly count, top 8 anomalous weeks with holiday attribution"],
        ["AI Recommendations",          "5 actionable business recommendations"],
        ["Methodology Note",            "Cleaning decisions, split strategy, model selection rationale"],
    ]
    struct_df = pd.DataFrame(structure, columns=["Section", "Contents"])
    st.dataframe(struct_df, use_container_width=True, hide_index=True)

    # ── Cleaning decisions appendix ────────────────────────────────────────────
    with st.expander("📎 Data Cleaning Decisions (Appendix)"):
        for entry in log:
            st.markdown(f"- {entry}")
