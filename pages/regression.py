"""pages/regression.py — Complete Regression Analysis Page"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from modules.data_loader import load_and_clean_data
from modules.regression import train_all_models, predict_single, FEATURE_COLS
from utils.charts import actual_vs_predicted, coefficient_bar, vif_chart, LAYOUT
from utils.helpers import r2_badge, mape_badge, info_card, fmt_currency


@st.cache_resource(show_spinner=False)
def cached_models():
    df, _ = load_and_clean_data()
    return train_all_models(df), df


def render():
    st.markdown('<div class="section-header">📈 Regression Analysis</div>', unsafe_allow_html=True)
    info_card(
        "Model Architecture",
        "Three linear models are trained on 26 engineered features. "
        "OLS is the interpretable baseline. Ridge (L2) shrinks correlated markdown "
        "coefficients. Lasso (L1) performs implicit feature selection — useful for identifying "
        "the minimal set of predictors. All use 80/20 chronological train/test split.",
        "#3b82f6",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Training models (cached after first run)..."):
        results, df = cached_models()

    model_names = list(results.keys())
    sel_model   = st.selectbox("Select model", model_names)
    res         = results[sel_model]
    m           = res["metrics"]

    # ── Metrics strip ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R²",       r2_badge(m.get("R²", 0)))
    c2.metric("Adj R²",   f"{m.get('Adj R²', 0):.4f}")
    c3.metric("RMSE",     fmt_currency(m.get("RMSE", 0)))
    c4.metric("MAE",      fmt_currency(m.get("MAE", 0)))
    c5.metric("MAPE",     mape_badge(m.get("MAPE (%)", 0)))

    # ── Comparison table ───────────────────────────────────────────────────────
    st.markdown("### 🏆 Model Comparison")
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        row.update(r["metrics"])
        rows.append(row)
    comp_df = pd.DataFrame(rows)

    def color_r2(val):
        try:
            v = float(val)
            if v >= 0.85: return "background-color: #14532d; color: #86efac"
            if v >= 0.70: return "background-color: #713f12; color: #fde68a"
            return "background-color: #450a0a; color: #fca5a5"
        except: return ""

    styled = comp_df.set_index("Model").style.applymap(color_r2, subset=["R²", "Adj R²"])
    st.dataframe(styled, use_container_width=True)

    st.caption(
        "Interpretation: R² = variance explained. RMSE penalises large errors "
        "(important for holiday spikes). MAPE is scale-free. "
        "Ridge typically outperforms OLS when markdown features are correlated (VIF > 5)."
    )

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Actual vs Predicted",
        "📊 Coefficients",
        "🔗 Multicollinearity (VIF)",
        "📐 Statistical Summary",
        "🔮 Manual Prediction",
    ])

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(
                actual_vs_predicted(res["y_test"], res["y_pred"], sel_model),
                use_container_width=True,
            )
        with col_r:
            residuals = np.array(res["y_test"]) - res["y_pred"]
            fig_res = px.histogram(
                residuals, nbins=60,
                color_discrete_sequence=["#8b5cf6"],
                title="Residuals Distribution",
                marginal="box",
                labels={"value": "Residual ($)"},
            )
            fig_res.update_layout(**LAYOUT, height=420)
            st.plotly_chart(fig_res, use_container_width=True)

        # Residuals over time
        y_test_arr = np.array(res["y_test"])
        residuals_df = pd.DataFrame({
            "Index":    range(len(residuals)),
            "Residual": residuals,
            "Actual":   y_test_arr,
        })
        fig_rt = px.scatter(
            residuals_df, x="Index", y="Residual",
            color="Actual", color_continuous_scale="RdBu",
            opacity=0.5,
            title="Residuals vs Observation Index (time order)",
        )
        fig_rt.add_hline(y=0, line_dash="dash", line_color="#ef4444")
        fig_rt.update_layout(**LAYOUT, height=380)
        st.plotly_chart(fig_rt, use_container_width=True)
        st.caption(
            "Homoscedastic residuals (random scatter) indicate a well-specified model. "
            "Funnel patterns indicate heteroscedasticity — common in retail sales data "
            "due to high variance in holiday weeks."
        )

    with tab2:
        n_show = st.slider("Features to display", 5, len(res["coef_df"]), 20, key="coef_slider")
        st.plotly_chart(coefficient_bar(res["coef_df"], top_n=n_show), use_container_width=True)

        with st.expander("📋 Full Coefficient Table"):
            coef_display = res["coef_df"].copy()
            coef_display["Abs Magnitude"] = coef_display["Coefficient"].abs()
            coef_display["Direction"] = coef_display["Coefficient"].apply(
                lambda x: "🟢 Positive" if x > 0 else "🔴 Negative"
            )
            st.dataframe(coef_display.round(4), use_container_width=True)

        st.markdown("""
        **Reading coefficients** (all features are standardised before fitting):
        - A coefficient of +5,000 means: *holding everything else fixed, a 1-SD increase
          in this feature is associated with $5,000 higher weekly sales.*
        - For Ridge/Lasso, coefficients are shrunk toward zero.
        - Lasso drives small coefficients exactly to zero (implicit feature selection).
        """)

    with tab3:
        st.markdown("""
        **Variance Inflation Factor** measures multicollinearity.
        Formula: VIF_j = 1 / (1 − R²_j) where R²_j is the R² from regressing
        feature j on all other features.
        """)
        info_card("Expected High VIF", 
                  "MarkDown1-5, Total_MarkDown, and seasonal time features will have "
                  "high VIF because they are intentionally correlated. "
                  "This is why Ridge regression (L2 penalty) is recommended over OLS.",
                  "#f59e0b")
        st.plotly_chart(vif_chart(res["vif_df"]), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**VIF > 10 (Problematic)**")
            high = res["vif_df"][res["vif_df"]["VIF"] > 10]
            st.dataframe(high, use_container_width=True)
        with col_r:
            st.markdown("**VIF 5–10 (Moderate)**")
            mod = res["vif_df"][(res["vif_df"]["VIF"] > 5) & (res["vif_df"]["VIF"] <= 10)]
            st.dataframe(mod, use_container_width=True)

    with tab4:
        if sel_model == "OLS (Multiple Linear Regression)" and "sm_summary" in res:
            st.markdown("**Statsmodels OLS Full Summary** (p-values, t-statistics, confidence intervals)")
            st.code(res["sm_summary"].summary().as_text(), language=None)
            st.markdown("""
            **Key statistics to check:**
            - **Prob (F-statistic)** < 0.05 → model is statistically significant overall.
            - **Individual p-values** < 0.05 → that feature significantly predicts sales.
            - **Durbin-Watson** ≈ 2 → no autocorrelation in residuals.
            - **Jarque-Bera** → normality of residuals (violated in retail data; robust SE preferred).
            - **Cond. No.** > 1000 → severe multicollinearity flagged.
            """)
        else:
            st.info(f"Detailed statsmodels summary only available for OLS. "
                    f"Switch to 'OLS (Multiple Linear Regression)' above.")

    with tab5:
        st.markdown("### 🔮 Manual Sales Prediction")
        st.caption("Adjust feature values to get a live prediction from the selected model.")
        col_l, col_r = st.columns(2)
        user_inputs = {}
        feature_defaults = {
            "Temperature": 60.0, "Fuel_Price": 3.2, "CPI": 220.0,
            "Unemployment": 8.0, "Total_MarkDown": 2000.0,
            "Size": 150000.0, "IsHoliday": 0.0, "Week": 26.0,
            "Month": 6.0, "Year": 2012.0, "Store_Mean_Sales": 18000.0,
            "Dept_Mean_Sales": 15000.0, "Sales_Roll_Mean_4w": 16000.0,
            "Sales_Roll_Mean_12w": 17000.0,
        }
        available_feats = [f for f in res["feature_names"] if f in feature_defaults]
        half = len(available_feats) // 2
        for i, feat in enumerate(available_feats):
            col = col_l if i < half else col_r
            user_inputs[feat] = col.number_input(
                feat, value=feature_defaults.get(feat, 0.0),
                format="%.2f", key=f"pred_{feat}"
            )

        if st.button("Predict Weekly Sales", type="primary"):
            pred = predict_single(res, user_inputs)
            st.success(f"### 💰 Predicted Weekly Sales: {fmt_currency(pred)}")
