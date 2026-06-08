"""pages/xai.py — Complete Explainable AI Page"""

import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import load_and_clean_data
from modules.regression import train_all_models, FEATURE_COLS
from modules.xai import (
    compute_shap_values, shap_feature_importance,
    shap_waterfall_data, generate_narrative, partial_dependence_data,
    SHAP_AVAILABLE,
)
from utils.charts import shap_bar_chart, waterfall_chart, pdp_plot, LAYOUT
from utils.helpers import info_card, fmt_currency
import plotly.express as px
import plotly.graph_objects as go


@st.cache_resource(show_spinner=False)
def get_models_and_data():
    df, _ = load_and_clean_data()
    results = train_all_models(df)
    return results, df


def render():
    st.markdown('<div class="section-header">🧠 Explainable AI (XAI)</div>', unsafe_allow_html=True)
    info_card(
        "SHAP — SHapley Additive exPlanations",
        "SHAP decomposes each prediction into additive feature contributions using "
        "game-theoretic Shapley values. This satisfies four axioms: Efficiency (contributions sum to "
        "prediction), Symmetry (equal features get equal credit), Dummy (zero-impact features get 0), "
        "and Additivity. LinearExplainer provides exact (not approximate) SHAP values for linear models.",
        "#8b5cf6",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if not SHAP_AVAILABLE:
        st.error("SHAP not installed. Run: `pip install shap`")
        info_card("Install SHAP", "pip install shap==0.45.0", "#ef4444")
        return

    with st.spinner("Loading models (cached)..."):
        results, df = get_models_and_data()

    sel_model = st.selectbox(
        "Select model for explanation",
        ["OLS (Multiple Linear Regression)", "Ridge Regression (α=10)", "Lasso Regression (α=1)"],
    )
    res = results[sel_model]

    n_sample = st.slider("Sample size for SHAP computation", 100, 500, 300, 50)

    with st.spinner("Computing SHAP values..."):
        shap_vals, X_sample = compute_shap_values(res, df, n_sample=n_sample)

    if shap_vals is None:
        st.error("SHAP computation failed.")
        return

    imp_df = shap_feature_importance(shap_vals, res["feature_names"])

    # Store in session state
    st.session_state.update({
        "shap_vals": shap_vals,
        "X_sample": X_sample,
        "imp_df": imp_df,
        "res_for_xai": res,
    })

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Global Importance",
        "🔬 Local Explanation",
        "📈 Partial Dependence",
        "📝 Business Narrative",
    ])

    # ── Tab 1: Global ──────────────────────────────────────────────────────────
    with tab1:
        col_l, col_r = st.columns([3, 2])
        with col_l:
            top_n = st.slider("Features to display", 5, len(imp_df), 15)
            st.plotly_chart(shap_bar_chart(imp_df, top_n=top_n), use_container_width=True)

        with col_r:
            st.markdown("**Ranked Feature Importance**")
            display_imp = imp_df.head(top_n).copy()
            display_imp.index = range(1, len(display_imp) + 1)
            display_imp.index.name = "Rank"
            display_imp["Importance"] = display_imp["Importance"].map("${:,.2f}".format)
            st.dataframe(display_imp, use_container_width=True)

        st.markdown("""
        **Interpretation:**
        Mean |SHAP value| = average absolute impact on predicted sales across all samples.
        A feature with importance $5,000 shifts predictions by $5,000 on average.

        **Key findings (typical for Walmart data):**
        - `Store_Mean_Sales` and `Dept_Mean_Sales` dominate → store/dept identity is the strongest predictor.
        - `Is_Thanksgiving` and `Is_Christmas` → holiday effects are individually large.
        - `Total_MarkDown` → promotional spend matters but less than store baseline.
        - `Unemployment` and `CPI` → macro-economic features have moderate impact.
        """)

        # SHAP heatmap: top features × samples
        top_feats = imp_df["Feature"].head(10).tolist()
        feat_indices = [res["feature_names"].index(f) for f in top_feats if f in res["feature_names"]]
        if feat_indices:
            heat_data = shap_vals[:, feat_indices]
            fig_heat = px.imshow(
                heat_data.T,
                x=[f"Sample {i}" for i in range(heat_data.shape[0])],
                y=top_feats,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="SHAP Values Heatmap (top 10 features × samples)",
            )
            fig_heat.update_layout(**LAYOUT, height=420)
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Tab 2: Local ───────────────────────────────────────────────────────────
    with tab2:
        st.markdown("Select a sample observation to see *why* the model made that specific prediction.")
        row_idx = st.slider("Observation index", 0, shap_vals.shape[0] - 1, 0)

        # Compute base value and prediction
        scaler = res["scaler"]
        model  = res["model"]
        avail  = [c for c in FEATURE_COLS if c in df.columns]
        subset = df[avail].dropna().sample(n_sample, random_state=42)
        X_sc   = scaler.transform(subset)
        all_preds = model.predict(X_sc)
        base_value = float(all_preds.mean())
        this_pred  = float(all_preds[row_idx])

        col_l, col_r = st.columns(2)
        col_l.metric("Base Value (avg prediction)", fmt_currency(base_value))
        col_r.metric("This Prediction", fmt_currency(this_pred),
                     delta=fmt_currency(this_pred - base_value))

        wf_df = shap_waterfall_data(shap_vals, res["feature_names"], base_value, row_idx)
        st.plotly_chart(waterfall_chart(wf_df, base_value), use_container_width=True)

        st.markdown("**Top feature contributions for this prediction:**")
        wf_display = wf_df[["Feature", "SHAP_Value", "Direction"]].head(10).copy()
        wf_display["SHAP_Value"] = wf_display["SHAP_Value"].map("${:,.2f}".format)
        st.dataframe(wf_display, use_container_width=True, hide_index=True)

    # ── Tab 3: PDP ─────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        **Partial Dependence Plot** shows the marginal effect of one feature on predicted sales,
        averaging over all other features' observed distributions.
        It answers: *"How does changing X affect predicted sales on average?"*
        """)
        num_feats = [f for f in res["feature_names"]
                     if f in df.columns and df[f].dtype in ["float64", "int64", "int32"]]

        col_l, col_r = st.columns(2)
        with col_l:
            feat1 = st.selectbox("Feature 1", num_feats,
                                  index=num_feats.index("Total_MarkDown") if "Total_MarkDown" in num_feats else 0)
        with col_r:
            feat2 = st.selectbox("Feature 2", num_feats,
                                  index=num_feats.index("Unemployment") if "Unemployment" in num_feats else 1)

        grid_size = st.slider("Grid resolution", 20, 60, 40)

        col_p, col_q = st.columns(2)
        for col, feat in [(col_p, feat1), (col_q, feat2)]:
            with col:
                with st.spinner(f"Computing PDP for {feat}..."):
                    pdp_df = partial_dependence_data(res, df, feat, n_grid=grid_size)
                if not pdp_df.empty:
                    st.plotly_chart(pdp_plot(pdp_df, feat), use_container_width=True)
                    # Trend interpretation
                    slope = np.polyfit(pdp_df["Feature_Value"], pdp_df["Predicted_Sales"], 1)[0]
                    direction = "positive" if slope > 0 else "negative"
                    st.caption(f"Trend: {direction} ({slope:+,.1f} $/unit increase in {feat})")

    # ── Tab 4: Narrative ───────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 💬 AI Business Narrative")
        st.caption("Auto-generated interpretation of SHAP analysis for non-technical stakeholders.")

        mean_shap = shap_vals.mean(axis=0)
        wf_mean = pd.DataFrame({
            "Feature":    res["feature_names"],
            "SHAP_Value": mean_shap,
        })

        avail  = [c for c in FEATURE_COLS if c in df.columns]
        subset = df[avail].dropna().sample(min(200, len(df)), random_state=1)
        X_sc   = res["scaler"].transform(subset)
        avg_pred = float(res["model"].predict(X_sc).mean())

        narrative = generate_narrative(wf_mean, avg_pred)

        st.markdown(
            f"""
            <div style="background:#1e293b; border:1px solid #334155; border-radius:10px;
                        padding:20px; color:#e2e8f0; font-size:0.95rem; line-height:1.7;">
            {narrative.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("""
        **How to use this in a business context:**
        1. Present the top upward drivers to justify marketing spend (e.g. "markdowns are working").
        2. Present downward pressures to leadership to flag risks (e.g. "unemployment is a headwind").
        3. Use local explanations (Tab 2) to explain individual store/week predictions to store managers.
        4. Use PDPs (Tab 3) to set optimal markdown levels — find the inflection point.
        """)
