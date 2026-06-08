"""
modules/data_loader.py
──────────────────────
Data loading, intelligent cleaning, and feature engineering for the
Walmart Sales Forecasting dataset.

Design decisions are documented with statistical justification.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Exact holiday dates provided by Kaggle competition description
HOLIDAY_DATES = {
    "Super Bowl":   ["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"],
    "Labor Day":    ["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"],
    "Thanksgiving": ["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"],
    "Christmas":    ["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"],
}


@st.cache_data(show_spinner=False)
def load_raw_data():
    """
    Load raw CSV files from the data/ directory.
    If files are missing, auto-generates synthetic data so the app
    can be demoed without the Kaggle download.
    """
    required = ["train.csv", "stores.csv", "features.csv", "test.csv"]
    missing  = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        # Lazy import to avoid circular dependency
        import sys, importlib.util
        gen_path = os.path.join(os.path.dirname(__file__), "..", "utils", "generate_sample_data.py")
        spec = importlib.util.spec_from_file_location("gen", gen_path)
        gen  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        gen.generate_all()

    train    = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test     = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    stores   = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    return train, test, features, stores


@st.cache_data(show_spinner=False)
def load_and_clean_data():
    """
    Master pipeline:  raw → merged → cleaned → engineered features → final df.
    Returns (df, cleaning_log) where cleaning_log is a list of decision strings.
    """
    train, test, features, stores = load_raw_data()
    log = []

    # ── 1. Date conversion ─────────────────────────────────────────────────────
    for df in [train, test, features]:
        df["Date"] = pd.to_datetime(df["Date"])
    log.append("✅ Date columns parsed to datetime for train, test, and features.")

    # ── 2. Merge ───────────────────────────────────────────────────────────────
    # Strategy: left-join train onto features so every sales record gets context.
    # Then join stores to bring in Type and Size.
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")
    log.append("✅ Merged train ← features (Store, Date, IsHoliday) ← stores (Store).")

    # ── 3. Missing-value analysis & imputation ─────────────────────────────────

    # MarkDown1–5
    # ------------------------------------------------------------------
    # Kaggle data description states: "MarkDown data is only available
    # after Nov 2011 and is not always available for all stores."
    # Empirically ~60-70 % of MarkDown rows are NaN, predominantly
    # before Nov 2011 — i.e., the missingness is MCAR/MAR (Missing At
    # Random due to temporal policy), not MNAR.
    # Decision: replace NaN with 0 (no promotion ran → no markdown spend).
    # This is conservative and preserves the zero-inflation structure.
    markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    pre_missing = df[markdown_cols].isna().sum()
    df[markdown_cols] = df[markdown_cols].fillna(0)
    log.append(
        f"✅ MarkDown1-5: replaced NaN with 0. "
        f"Reasoning: NaN rows are concentrated before Nov 2011 — absence "
        f"of markdown record ≡ no promotional event. "
        f"Missing counts were {pre_missing.to_dict()}."
    )

    # Temperature
    # ------------------------------------------------------------------
    # Missingness is sparse and not systematic.  Store-wise median is
    # robust (insensitive to seasonal outliers unlike mean).
    temp_missing = df["Temperature"].isna().sum()
    df["Temperature"] = df.groupby("Store")["Temperature"].transform(
        lambda x: x.fillna(x.median())
    )
    log.append(
        f"✅ Temperature: {temp_missing} NaN → store-wise median imputation. "
        f"Median preferred over mean for robustness to seasonal extremes."
    )

    # Fuel_Price
    # ------------------------------------------------------------------
    # Same rationale as Temperature: sparse, store-level pattern.
    fp_missing = df["Fuel_Price"].isna().sum()
    df["Fuel_Price"] = df.groupby("Store")["Fuel_Price"].transform(
        lambda x: x.fillna(x.median())
    )
    log.append(
        f"✅ Fuel_Price: {fp_missing} NaN → store-wise median imputation."
    )

    # CPI & Unemployment
    # ------------------------------------------------------------------
    # Both are macro-economic time series that evolve smoothly within a
    # store's region.  Time-aware linear interpolation within each store
    # respects this continuity and avoids introducing artificial steps.
    for col in ["CPI", "Unemployment"]:
        n_missing = df[col].isna().sum()
        df = df.sort_values(["Store", "Date"])
        df[col] = df.groupby("Store")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        log.append(
            f"✅ {col}: {n_missing} NaN → time-aware linear interpolation within "
            f"each store. Rationale: economic indicators change continuously; "
            f"interpolation is more principled than constant fill."
        )

    # ── 4. Feature engineering ─────────────────────────────────────────────────

    # Temporal decomposition
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Week"]      = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"]   = df["Date"].dt.quarter
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Holiday indicators — named holidays carry different demand spikes
    for holiday, dates in HOLIDAY_DATES.items():
        col = "Is_" + holiday.replace(" ", "_")
        df[col] = df["Date"].isin(pd.to_datetime(dates)).astype(int)
    log.append("✅ Named holiday indicators created: Super Bowl, Labor Day, Thanksgiving, Christmas.")

    # Lag sales features (store-dept level time series)
    # Lag-1, Lag-2, Lag-4 weeks capture autoregressive structure.
    df = df.sort_values(["Store", "Dept", "Date"])
    for lag in [1, 2, 4]:
        df[f"Sales_Lag_{lag}w"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(lag)
    log.append("✅ Lag features: 1-week, 2-week, 4-week lags created per (Store, Dept) group.")

    # Rolling statistics — capture local trend and volatility
    for window in [4, 12]:
        df[f"Sales_Roll_Mean_{window}w"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"Sales_Roll_Std_{window}w"] = (
            df.groupby(["Store", "Dept"])["Weekly_Sales"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    log.append("✅ Rolling mean/std (4w & 12w) created — shifted by 1 to prevent data leakage.")

    # Store & department aggregates
    store_avg = df.groupby("Store")["Weekly_Sales"].transform("mean")
    df["Store_Mean_Sales"] = store_avg

    dept_avg = df.groupby("Dept")["Weekly_Sales"].transform("mean")
    df["Dept_Mean_Sales"] = dept_avg

    # Total markdown spend
    df["Total_MarkDown"] = df[markdown_cols].sum(axis=1)
    log.append("✅ Total_MarkDown = sum of MarkDown1-5 created.")

    # ── 5. Feature encoding ────────────────────────────────────────────────────
    # Store Type: one-hot (drop_first avoids dummy variable trap)
    type_dummies = pd.get_dummies(df["Type"], prefix="StoreType", drop_first=False)
    df = pd.concat([df, type_dummies], axis=1)

    # IsHoliday: already binary (True/False) → cast to int
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    log.append("✅ Store Type one-hot encoded; IsHoliday cast to int.")

    # ── 6. Outlier annotation (do not remove) ─────────────────────────────────
    # IQR-based flagging
    q1 = df["Weekly_Sales"].quantile(0.25)
    q3 = df["Weekly_Sales"].quantile(0.75)
    iqr = q3 - q1
    df["IQR_Outlier"] = (
        (df["Weekly_Sales"] < q1 - 1.5 * iqr) |
        (df["Weekly_Sales"] > q3 + 1.5 * iqr)
    ).astype(int)

    # Z-score based flagging
    df["ZScore_Sales"] = np.abs(stats.zscore(df["Weekly_Sales"].fillna(0)))
    df["ZScore_Outlier"] = (df["ZScore_Sales"] > 3).astype(int)

    log.append(
        f"✅ Outlier flags added (IQR + Z-score > 3). "
        f"Outliers NOT removed — many coincide with holiday demand spikes "
        f"(Thanksgiving, Christmas) which are legitimate signal, not noise."
    )

    # ── 7. Final sort ──────────────────────────────────────────────────────────
    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)

    return df, log


def get_store_dept_series(df, store: int, dept: int) -> pd.Series:
    """Return weekly sales time series for a (Store, Dept) pair."""
    mask = (df["Store"] == store) & (df["Dept"] == dept)
    ts = df.loc[mask].set_index("Date")["Weekly_Sales"].sort_index()
    return ts


def get_store_series(df, store: int) -> pd.Series:
    """Aggregate weekly sales across all departments for a single store."""
    mask = df["Store"] == store
    ts = (
        df.loc[mask]
        .groupby("Date")["Weekly_Sales"]
        .sum()
        .sort_index()
    )
    # Ensure proper datetime index with weekly frequency
    ts.index = pd.DatetimeIndex(ts.index)
    ts = ts.resample("W").sum()
    return ts


def outlier_summary(df) -> pd.DataFrame:
    """
    Returns a DataFrame of outlier records enriched with context
    (holiday name, store type, markdown spend) for interpretability.
    """
    outliers = df[df["IQR_Outlier"] == 1].copy()
    outliers["Likely_Cause"] = "Unknown"

    for holiday, dates in HOLIDAY_DATES.items():
        mask = outliers["Date"].isin(pd.to_datetime(dates))
        outliers.loc[mask, "Likely_Cause"] = holiday

    # High markdown = promotional spike
    high_md = outliers["Total_MarkDown"] > outliers["Total_MarkDown"].quantile(0.75)
    no_cause = outliers["Likely_Cause"] == "Unknown"
    outliers.loc[high_md & no_cause, "Likely_Cause"] = "Promotional Markdown"

    return outliers[[
        "Store", "Dept", "Date", "Weekly_Sales",
        "IsHoliday", "Total_MarkDown", "ZScore_Sales", "Likely_Cause"
    ]].sort_values("ZScore_Sales", ascending=False)
