"""
modules/forecasting.py
──────────────────────
ARIMA and SARIMA forecasting for Walmart store-level weekly sales.

Statistical decisions:
  - Stationarity: Augmented Dickey–Fuller test.  If p > 0.05, apply
    first-order differencing (d=1).  Seasonal differencing (D=1) added
    for SARIMA when annual periodicity is detected.
  - Order selection: minimise AIC over a parameter grid (p,d,q in 0-2).
    For SARIMA the seasonal period m=52 (weekly data, annual cycle).
  - Confidence intervals: 95 % prediction intervals propagated from the
    ARIMA state-space representation (default in statsmodels).
  - Forecast horizons: 4 weeks (short), 12 weeks (quarter), 52 weeks (year).
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

warnings.filterwarnings("ignore")

HORIZONS = {"4 weeks": 4, "12 weeks": 12, "52 weeks": 52}


# ── Stationarity ───────────────────────────────────────────────────────────────

def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey–Fuller test.
    H₀: the series has a unit root (non-stationary).
    Reject H₀ at α=0.05 → series is stationary.
    """
    series = series.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 10:
        return {"ADF Statistic": np.nan, "p-value": 1.0,
                "Critical Values": {}, "Stationary": False}
    result = adfuller(series, autolag="AIC")
    return {
        "ADF Statistic": round(result[0], 4),
        "p-value":       round(result[1], 4),
        "Critical Values": {k: round(v, 4) for k, v in result[4].items()},
        "Stationary":    result[1] < 0.05,
    }


def make_stationary(series: pd.Series):
    """
    Apply minimum differencing to achieve stationarity.
    Returns (transformed_series, d_order, log_transformed).
    """
    # Guard: log-transform if strictly positive (stabilises variance)
    log_transformed = False
    if (series > 0).all():
        series = np.log(series)
        log_transformed = True

    d = 0
    for _ in range(2):  # max d=2
        if adf_test(series)["Stationary"]:
            break
        series = series.diff().dropna()
        d += 1

    return series, d, log_transformed


# ── ACF / PACF helpers ─────────────────────────────────────────────────────────

def compute_acf_pacf(series: pd.Series, nlags: int = 40):
    series = series.dropna()
    acf_vals  = acf(series, nlags=nlags)
    pacf_vals = pacf(series, nlags=nlags)
    return acf_vals, pacf_vals


# ── Order selection ────────────────────────────────────────────────────────────

def _best_arima_order(series: pd.Series, d: int):
    """
    Grid search over p, q ∈ {0,1,2} to minimise AIC.
    d is fixed from the stationarity test.
    """
    best_aic, best_order = np.inf, (1, d, 1)
    for p, q in itertools.product(range(3), range(3)):
        try:
            m = ARIMA(series, order=(p, d, q)).fit()
            if m.aic < best_aic:
                best_aic   = m.aic
                best_order = (p, d, q)
        except Exception:
            continue
    return best_order, best_aic


def _best_sarima_order(series: pd.Series, d: int, m: int = 52):
    """
    Grid search SARIMA(p,d,q)(P,D,Q,m).
    Restricted to small parameter space for computational feasibility.
    Seasonal period m=52 for weekly annual seasonality.
    """
    best_aic, best_order, best_seasonal = np.inf, (1, d, 1), (1, 1, 1, m)
    for p, q, P, Q in itertools.product(range(2), range(2), range(2), range(2)):
        try:
            mod = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=(P, 1, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            if mod.aic < best_aic:
                best_aic      = mod.aic
                best_order    = (p, d, q)
                best_seasonal = (P, 1, Q, m)
        except Exception:
            continue
    return best_order, best_seasonal, best_aic


# ── Metrics ────────────────────────────────────────────────────────────────────

def _ts_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mask = np.array(y_true) != 0
    mape = np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask])
                           / np.array(y_true)[mask])) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE (%)": round(mape, 2)}


# ── ARIMA ──────────────────────────────────────────────────────────────────────

def fit_arima(series: pd.Series, horizon_label: str = "12 weeks"):
    """
    Fit ARIMA model on a weekly sales series.
    Returns forecast dict with values, conf intervals, metrics, and order info.
    """
    horizon = HORIZONS.get(horizon_label, 12)
    series  = series.dropna()
    series.index = pd.DatetimeIndex(series.index)
    series  = series.resample("W").sum().replace([np.inf, -np.inf], np.nan).ffill().dropna()

    # Stationarity
    adf_result = adf_test(series)
    d = 0 if adf_result["Stationary"] else 1

    # Order selection (on train portion)
    train_size = int(len(series) * 0.85)
    train = series.iloc[:train_size]
    test  = series.iloc[train_size:]

    order, aic = _best_arima_order(train, d)

    # Refit on full series
    model = ARIMA(series, order=order).fit()

    # Out-of-sample forecast
    forecast_result = model.get_forecast(steps=horizon)
    fc_mean  = forecast_result.predicted_mean
    fc_ci    = forecast_result.conf_int(alpha=0.05)

    # In-sample backtest metrics
    fitted = model.fittedvalues.iloc[train_size:]
    actual = series.iloc[train_size:]
    metrics = _ts_metrics(actual, fitted) if len(actual) > 0 else {}

    # Build future date index
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1),
                                  periods=horizon, freq="W-FRI")
    fc_df = pd.DataFrame({
        "Date":     future_dates,
        "Forecast": fc_mean.values,
        "Lower_CI": fc_ci.iloc[:, 0].values,
        "Upper_CI": fc_ci.iloc[:, 1].values,
    })

    return {
        "model":      model,
        "order":      order,
        "aic":        round(aic, 2),
        "adf":        adf_result,
        "d":          d,
        "forecast":   fc_df,
        "historical": series,
        "metrics":    metrics,
        "model_type": "ARIMA",
    }


# ── SARIMA ─────────────────────────────────────────────────────────────────────

def fit_sarima(series: pd.Series, horizon_label: str = "12 weeks", m: int = 52):
    """
    Fit SARIMA model.
    m=52 captures annual weekly seasonality (e.g. Christmas uplift repeats yearly).
    """
    horizon = HORIZONS.get(horizon_label, 12)
    series  = series.dropna()
    series.index = pd.DatetimeIndex(series.index)
    series  = series.resample("W").sum().replace([np.inf, -np.inf], np.nan).ffill().dropna()

    adf_result = adf_test(series)
    d = 0 if adf_result["Stationary"] else 1

    train_size = int(len(series) * 0.85)
    train = series.iloc[:train_size]

    order, seasonal_order, aic = _best_sarima_order(train, d, m=m)

    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    forecast_result = model.get_forecast(steps=horizon)
    fc_mean = forecast_result.predicted_mean
    fc_ci   = forecast_result.conf_int(alpha=0.05)

    last_date    = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(weeks=1),
                                  periods=horizon, freq="W-FRI")
    fc_df = pd.DataFrame({
        "Date":     future_dates,
        "Forecast": fc_mean.values,
        "Lower_CI": fc_ci.iloc[:, 0].values,
        "Upper_CI": fc_ci.iloc[:, 1].values,
    })

    # Backtest on held-out tail
    fitted  = model.fittedvalues.iloc[train_size:]
    actual  = series.iloc[train_size:]
    metrics = _ts_metrics(actual, fitted) if len(actual) > 0 else {}

    return {
        "model":          model,
        "order":          order,
        "seasonal_order": seasonal_order,
        "aic":            round(aic, 2),
        "adf":            adf_result,
        "d":              d,
        "forecast":       fc_df,
        "historical":     series,
        "metrics":        metrics,
        "model_type":     "SARIMA",
    }


# ── Comparison helper ──────────────────────────────────────────────────────────

def compare_models(arima_result: dict, sarima_result: dict) -> pd.DataFrame:
    """Side-by-side model comparison table."""
    rows = []
    for res in [arima_result, sarima_result]:
        row = {"Model": res["model_type"], "AIC": res["aic"]}
        row.update(res["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)
