"""
modules/xai.py
──────────────
Explainable AI using SHAP for regression models.
Generates:
  - Global feature importance (SHAP bar plot data)
  - Per-prediction waterfall explanation
  - Natural language business narrative
  - Partial dependence data for key features
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ── SHAP computation ───────────────────────────────────────────────────────────

def compute_shap_values(model_result: dict, df: pd.DataFrame, n_sample: int = 300):
    """
    Compute SHAP values for the given sklearn model.
    Uses LinearExplainer (exact, fast) for linear models.
    Returns (shap_values array, X_sample DataFrame).
    """
    if not SHAP_AVAILABLE:
        return None, None

    from modules.regression import FEATURE_COLS
    available = [c for c in FEATURE_COLS if c in df.columns]
    subset    = df[available].dropna()

    # Sample for speed; SHAP values are stable across random subsamples
    X_sample = subset.sample(min(n_sample, len(subset)), random_state=42)

    scaler = model_result["scaler"]
    model  = model_result["model"]

    X_scaled = scaler.transform(X_sample)

    explainer   = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(X_scaled)

    return shap_values, X_sample


def shap_feature_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Mean |SHAP value| per feature → global importance ranking.
    """
    importances = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


def shap_waterfall_data(shap_values: np.ndarray, feature_names: list,
                         base_value: float, row_idx: int = 0) -> pd.DataFrame:
    """
    Waterfall chart data for a single prediction.
    Returns DataFrame with Feature, SHAP_Value, Direction.
    """
    sv = shap_values[row_idx]
    df = pd.DataFrame({
        "Feature":    feature_names,
        "SHAP_Value": sv,
    }).sort_values("SHAP_Value", key=abs, ascending=False)
    df["Direction"] = df["SHAP_Value"].apply(lambda x: "Positive ▲" if x > 0 else "Negative ▼")
    df["Base_Value"] = base_value
    return df.head(15)  # top 15 drivers


# ── Natural language narrative ─────────────────────────────────────────────────

def generate_narrative(shap_df: pd.DataFrame, prediction: float,
                        actual: float = None) -> str:
    """
    Convert top SHAP drivers into a boardroom-ready business narrative.
    """
    top_pos = shap_df[shap_df["SHAP_Value"] > 0].head(3)
    top_neg = shap_df[shap_df["SHAP_Value"] < 0].head(3)

    pos_parts = [_feature_to_text(r["Feature"], r["SHAP_Value"], "drove") for _, r in top_pos.iterrows()]
    neg_parts = [_feature_to_text(r["Feature"], r["SHAP_Value"], "reduced") for _, r in top_neg.iterrows()]

    narrative = f"**Predicted Weekly Sales: ${prediction:,.0f}**\n\n"

    if actual is not None:
        delta = prediction - actual
        narrative += f"**Actual Sales: ${actual:,.0f}** | Error: ${abs(delta):,.0f} "
        narrative += ("(over-forecast)\n\n" if delta > 0 else "(under-forecast)\n\n")

    if pos_parts:
        narrative += "**Key Upward Drivers:**\n"
        for p in pos_parts:
            narrative += f"  - {p}\n"

    if neg_parts:
        narrative += "\n**Key Downward Pressures:**\n"
        for p in neg_parts:
            narrative += f"  - {p}\n"

    return narrative


def _feature_to_text(feature: str, shap_val: float, verb: str) -> str:
    """Map feature name + SHAP value to human-readable sentence."""
    feat_map = {
        "Total_MarkDown":       "promotional markdown spend",
        "IsHoliday":            "holiday week effect",
        "Store_Mean_Sales":     "store baseline performance",
        "Dept_Mean_Sales":      "department baseline performance",
        "CPI":                  "consumer price index (inflation)",
        "Unemployment":         "regional unemployment rate",
        "Temperature":          "local temperature",
        "Fuel_Price":           "fuel prices",
        "Size":                 "store size",
        "Week":                 "week-of-year seasonality",
        "Month":                "monthly seasonality",
        "Is_Thanksgiving":      "Thanksgiving holiday effect",
        "Is_Christmas":         "Christmas holiday effect",
        "Is_Super_Bowl":        "Super Bowl week effect",
        "Is_Labor_Day":         "Labor Day effect",
        "Sales_Roll_Mean_4w":   "recent 4-week sales trend",
        "Sales_Roll_Mean_12w":  "quarterly sales trend",
        "StoreType_A":          "Type A store premium",
        "StoreType_B":          "Type B store effect",
    }
    label = feat_map.get(feature, feature.replace("_", " ").title())
    direction = "higher" if shap_val > 0 else "lower"
    return f"{label} {verb} sales {direction} (impact: ${abs(shap_val):,.0f})"


# ── Partial dependence helper ──────────────────────────────────────────────────

def partial_dependence_data(model_result: dict, df: pd.DataFrame,
                             feature: str, n_grid: int = 30) -> pd.DataFrame:
    """
    Compute partial dependence of predicted sales on a single feature.
    Uses the ice-cream approach: fix all other features at their median,
    vary the target feature across its observed range.
    """
    from modules.regression import FEATURE_COLS
    available = [c for c in FEATURE_COLS if c in df.columns]
    if feature not in available:
        return pd.DataFrame()

    subset    = df[available].dropna()
    medians   = subset.median()
    grid_vals = np.linspace(subset[feature].min(), subset[feature].max(), n_grid)

    preds = []
    scaler = model_result["scaler"]
    model  = model_result["model"]

    for val in grid_vals:
        row = medians.copy()
        row[feature] = val
        X = row[available].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        preds.append(float(model.predict(X_scaled)[0]))

    return pd.DataFrame({"Feature_Value": grid_vals, "Predicted_Sales": preds})
