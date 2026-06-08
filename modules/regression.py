"""
modules/regression.py
─────────────────────
Multiple Linear Regression, Ridge, and Lasso models for Weekly_Sales.
Includes:
  - Model training & evaluation (R², Adj-R², RMSE, MAE, MAPE)
  - Coefficient interpretation
  - Feature importance ranking
  - Multicollinearity diagnosis via VIF
  - Prediction API
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings, joblib, os

warnings.filterwarnings("ignore")

# Features used for regression
FEATURE_COLS = [
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "Total_MarkDown", "Size", "IsHoliday",
    "Year", "Month", "Week", "Quarter",
    "Is_Super_Bowl", "Is_Labor_Day", "Is_Thanksgiving", "Is_Christmas",
    "Store_Mean_Sales", "Dept_Mean_Sales",
    "Sales_Roll_Mean_4w", "Sales_Roll_Mean_12w",
    "StoreType_A", "StoreType_B", "StoreType_C",
]


def prepare_regression_data(df: pd.DataFrame):
    """
    Select, clean, and scale features for regression.
    Returns X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    subset = df[available + ["Weekly_Sales"]].dropna()

    X = subset[available]
    y = subset["Weekly_Sales"]

    # 80/20 chronological split — avoids future-leakage unlike random split.
    # Retail data has strong temporal dependencies; random splitting would
    # artificially inflate test performance via lag/rolling feature leakage.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, available


def _metrics(y_true, y_pred, n_features, n_samples) -> dict:
    r2   = r2_score(y_true, y_pred)
    adj  = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # MAPE: guard against division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"R²": round(r2, 4), "Adj R²": round(adj, 4),
            "RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE (%)": round(mape, 2)}


def train_all_models(df: pd.DataFrame):
    """
    Train OLS, Ridge (α=10), Lasso (α=1) on prepared regression data.
    Returns dict keyed by model name with sub-keys:
      model, metrics, coef_df, vif_df, y_test, y_pred, feature_names.
    """
    X_tr, X_te, y_tr, y_te, scaler, feat_names = prepare_regression_data(df)
    n, p = X_tr.shape

    results = {}

    model_specs = {
        "OLS (Multiple Linear Regression)": LinearRegression(),
        "Ridge Regression (α=10)": Ridge(alpha=10),
        "Lasso Regression (α=1)": Lasso(alpha=1, max_iter=5000),
    }

    for name, model in model_specs.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        # Coefficient dataframe
        coef_df = pd.DataFrame({
            "Feature":     feat_names,
            "Coefficient": model.coef_,
        }).sort_values("Coefficient", key=abs, ascending=False)

        # VIF — computed on unscaled training set to preserve interpretability
        # We recompute original (unscaled) data for VIF
        subset = df[[c for c in feat_names if c in df.columns]].dropna()
        split_idx = int(len(subset) * 0.8)
        X_raw = subset.iloc[:split_idx]
        vif_df = _compute_vif(X_raw, feat_names)

        results[name] = {
            "model":        model,
            "scaler":       scaler,
            "metrics":      _metrics(y_te, y_pred, p, len(y_te)),
            "coef_df":      coef_df,
            "vif_df":       vif_df,
            "y_test":       y_te,
            "y_pred":       y_pred,
            "feature_names": feat_names,
        }

    # Also fit statsmodels OLS for detailed statistical output
    X_tr_sm = sm.add_constant(X_tr)
    X_te_sm = sm.add_constant(X_te)
    ols_sm = sm.OLS(y_tr, X_tr_sm).fit()
    results["OLS (Multiple Linear Regression)"]["sm_summary"] = ols_sm

    return results


def _compute_vif(X_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Variance Inflation Factor per feature.
    VIF > 10 indicates problematic multicollinearity;
    VIF > 5 warrants attention.
    """
    X_clean = X_df.dropna()
    vif_data = []
    for i, col in enumerate(feature_names):
        if col not in X_clean.columns:
            continue
        try:
            vals = X_clean[feature_names].dropna().values
            v = variance_inflation_factor(vals, feature_names.index(col))
        except Exception:
            v = np.nan
        vif_data.append({"Feature": col, "VIF": round(v, 2)})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    vif_df["Severity"] = vif_df["VIF"].apply(
        lambda x: "🔴 High" if x > 10 else ("🟡 Moderate" if x > 5 else "🟢 OK")
    )
    return vif_df


def predict_single(model_result: dict, input_dict: dict) -> float:
    """
    Run inference on a manually constructed feature dict.
    Fills missing features with 0.
    """
    feat = model_result["feature_names"]
    scaler = model_result["scaler"]
    model  = model_result["model"]

    row = np.array([input_dict.get(f, 0.0) for f in feat]).reshape(1, -1)
    row_scaled = scaler.transform(row)
    return float(model.predict(row_scaled)[0])
