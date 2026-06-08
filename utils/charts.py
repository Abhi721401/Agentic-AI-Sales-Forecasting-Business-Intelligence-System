"""
utils/charts.py
───────────────
Reusable Plotly chart factory for the RetailGPT dashboard.
All charts use a consistent dark theme matching the Streamlit CSS.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Theme ──────────────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155"),
    colorway=["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6",
              "#06b6d4", "#ec4899", "#10b981"],
)

BLUE   = "#3b82f6"
GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"
PURPLE = "#8b5cf6"
CYAN   = "#06b6d4"


def apply_theme(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(title=title, height=height, **LAYOUT)
    return fig


# ── Sales trend ────────────────────────────────────────────────────────────────

def sales_trend(df: pd.DataFrame, store: int = None, dept: int = None,
                title: str = "Weekly Sales Trend") -> go.Figure:
    sub = df.copy()
    if store:
        sub = sub[sub["Store"] == store]
    if dept:
        sub = sub[sub["Dept"] == dept]

    agg = sub.groupby("Date")["Weekly_Sales"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["Date"], y=agg["Weekly_Sales"],
        mode="lines", name="Weekly Sales",
        line=dict(color=BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
    ))

    # Holiday markers
    holidays = sub[sub["IsHoliday"] == 1]["Date"].unique()
    holiday_sales = agg[agg["Date"].isin(holidays)]
    fig.add_trace(go.Scatter(
        x=holiday_sales["Date"], y=holiday_sales["Weekly_Sales"],
        mode="markers", name="Holiday Week",
        marker=dict(color=AMBER, size=7, symbol="diamond"),
    ))

    return apply_theme(fig, title)


def store_comparison_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    store_totals = (
        df.groupby("Store")["Weekly_Sales"].sum()
        .nlargest(top_n)
        .reset_index()
    )
    fig = px.bar(
        store_totals, x="Store", y="Weekly_Sales",
        color="Weekly_Sales", color_continuous_scale="Blues",
        labels={"Weekly_Sales": "Total Sales ($)"},
    )
    fig.update_coloraxes(showscale=False)
    return apply_theme(fig, f"Top {top_n} Stores by Total Revenue")


def dept_heatmap(df: pd.DataFrame, store: int = None) -> go.Figure:
    sub = df[df["Store"] == store] if store else df
    pivot = (
        sub.groupby(["Dept", "Month"])["Weekly_Sales"]
        .mean()
        .unstack(fill_value=0)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"M{c}" for c in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="Blues",
        hovertemplate="Dept %{y} | %{x}: $%{z:,.0f}<extra></extra>",
    ))
    return apply_theme(fig, "Department × Month Sales Heatmap", height=500)


# ── Regression ────────────────────────────────────────────────────────────────

def actual_vs_predicted(y_true, y_pred, model_name: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(y_true), y=list(y_pred),
        mode="markers",
        marker=dict(color=BLUE, opacity=0.5, size=4),
        name="Predictions",
    ))
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", line=dict(color=RED, dash="dash", width=1),
        name="Perfect Fit",
    ))
    return apply_theme(fig, f"{model_name} — Actual vs Predicted")


def coefficient_bar(coef_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top = coef_df.head(top_n).copy()
    colors = [GREEN if c > 0 else RED for c in top["Coefficient"]]
    fig = go.Figure(go.Bar(
        x=top["Coefficient"], y=top["Feature"],
        orientation="h",
        marker_color=colors,
    ))
    return apply_theme(fig, f"Top {top_n} Feature Coefficients", height=500)


def vif_chart(vif_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        vif_df.head(20), x="VIF", y="Feature",
        orientation="h",
        color="VIF",
        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
        range_color=[0, 20],
    )
    fig.add_vline(x=10, line_dash="dash", line_color=RED, annotation_text="VIF=10 threshold")
    fig.add_vline(x=5,  line_dash="dot",  line_color=AMBER, annotation_text="VIF=5")
    return apply_theme(fig, "Variance Inflation Factors (Multicollinearity Diagnosis)", height=480)


# ── Forecasting ───────────────────────────────────────────────────────────────

def forecast_plot(historical: pd.Series, fc_df: pd.DataFrame,
                  model_name: str = "Forecast") -> go.Figure:
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=historical.index, y=historical.values,
        mode="lines", name="Historical",
        line=dict(color=BLUE, width=2),
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
        y=pd.concat([fc_df["Upper_CI"], fc_df["Lower_CI"][::-1]]),
        fill="toself",
        fillcolor="rgba(139,92,246,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_df["Date"], y=fc_df["Forecast"],
        mode="lines+markers", name=f"{model_name} Forecast",
        line=dict(color=PURPLE, width=2, dash="dot"),
        marker=dict(size=5),
    ))

    return apply_theme(fig, f"{model_name} — Sales Forecast with Confidence Intervals", height=450)


def acf_pacf_plot(acf_vals, pacf_vals, nlags: int = 40) -> go.Figure:
    lags = list(range(len(acf_vals)))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"])

    conf = 1.96 / np.sqrt(nlags * 2)

    for col, vals, name in [(1, acf_vals, "ACF"), (2, pacf_vals, "PACF")]:
        fig.add_trace(go.Bar(x=lags, y=vals, name=name,
                             marker_color=BLUE, opacity=0.7), row=1, col=col)
        fig.add_hline(y=conf,  line_dash="dash", line_color=RED,   row=1, col=col)
        fig.add_hline(y=-conf, line_dash="dash", line_color=RED,   row=1, col=col)

    return apply_theme(fig, "ACF / PACF Analysis", height=380)


# ── Anomaly ───────────────────────────────────────────────────────────────────

def anomaly_scatter(df: pd.DataFrame, flag_col: str = "IF_Flag") -> go.Figure:
    normal = df[df[flag_col] == 0]
    anom   = df[df[flag_col] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["Date"], y=normal["Weekly_Sales"],
        mode="markers", name="Normal",
        marker=dict(color=BLUE, size=3, opacity=0.4),
    ))
    fig.add_trace(go.Scatter(
        x=anom["Date"], y=anom["Weekly_Sales"],
        mode="markers", name="Anomaly",
        marker=dict(color=RED, size=7, symbol="x"),
    ))
    return apply_theme(fig, "Anomaly Detection — Weekly Sales", height=440)


# ── XAI ──────────────────────────────────────────────────────────────────────

def shap_bar_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = importance_df.head(top_n)
    fig = px.bar(
        top, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_coloraxes(showscale=False)
    return apply_theme(fig, f"SHAP Feature Importance (Top {top_n})", height=450)


def waterfall_chart(waterfall_df: pd.DataFrame, base_value: float) -> go.Figure:
    measure = ["relative"] * len(waterfall_df) + ["total"]
    x_labels = list(waterfall_df["Feature"]) + ["Prediction"]
    y_values = list(waterfall_df["SHAP_Value"]) + [0]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=x_labels,
        y=y_values,
        base=base_value,
        connector={"line": {"color": "#334155"}},
        increasing={"marker": {"color": GREEN}},
        decreasing={"marker": {"color": RED}},
        totals={"marker": {"color": BLUE}},
    ))
    return apply_theme(fig, "SHAP Waterfall — Prediction Decomposition", height=480)


def pdp_plot(pdp_df: pd.DataFrame, feature: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=pdp_df["Feature_Value"], y=pdp_df["Predicted_Sales"],
        mode="lines+markers",
        line=dict(color=CYAN, width=2),
        marker=dict(size=5),
    ))
    return apply_theme(fig, f"Partial Dependence Plot — {feature}", height=380)
