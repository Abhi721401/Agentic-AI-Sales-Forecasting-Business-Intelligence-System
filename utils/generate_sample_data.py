"""
utils/generate_sample_data.py
──────────────────────────────
Generates realistic synthetic Walmart-style CSVs in data/ when the real
Kaggle files are not present.  This lets the app demo without authentication.

Run:  python utils/generate_sample_data.py
Or:   called automatically by data_loader.py when CSVs are missing.

Statistical properties preserved:
  - 45 stores, 3 types (A/B/C), realistic size distribution
  - 81 departments with heterogeneous sales levels
  - Weekly date grid 2010-02-05 → 2012-10-26
  - Holiday uplift (+15-40 % on holiday weeks)
  - Seasonal multiplicative pattern (peak Nov-Dec)
  - MarkDown data sparse before Nov 2011
  - AR(1) noise per (store, dept) track
  - CPI drift 211 → 230 over the period
  - Unemployment declining 9.8 → 7.8 %
"""

import os
import numpy as np
import pandas as pd

SEED = 42
rng  = np.random.default_rng(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Stores ─────────────────────────────────────────────────────────────────────
N_STORES = 45
STORE_TYPES = ["A"] * 22 + ["B"] * 17 + ["C"] * 6

def make_stores() -> pd.DataFrame:
    sizes = {
        "A": lambda: rng.integers(150_000, 220_000),
        "B": lambda: rng.integers(100_000, 150_000),
        "C": lambda: rng.integers(40_000,  100_000),
    }
    rows = []
    for s, t in enumerate(STORE_TYPES, 1):
        rows.append({"Store": s, "Type": t, "Size": sizes[t]()})
    return pd.DataFrame(rows)

# ── Date grid ──────────────────────────────────────────────────────────────────
DATES = pd.date_range("2010-02-05", "2012-10-26", freq="W-FRI")

HOLIDAY_DATES = pd.to_datetime([
    "2010-02-12","2011-02-11","2012-02-10",
    "2010-09-10","2011-09-09","2012-09-07",
    "2010-11-26","2011-11-25","2012-11-23",
    "2010-12-31","2011-12-30","2012-12-28",
])

# ── Features ───────────────────────────────────────────────────────────────────
def make_features(stores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_dates = len(DATES)

    for _, store_row in stores_df.iterrows():
        s = int(store_row["Store"])
        # Base temperature: store-specific mean + seasonal sine
        base_temp = rng.uniform(45, 75)
        temp_series = base_temp + 20 * np.sin(
            2 * np.pi * np.arange(n_dates) / 52
        ) + rng.normal(0, 3, n_dates)

        # Fuel price: slow drift with noise
        fuel = np.linspace(2.5, 3.8, n_dates) + rng.normal(0, 0.08, n_dates)

        # CPI: linear drift 211 → 230
        cpi = np.linspace(211, 230, n_dates) + rng.normal(0, 0.3, n_dates)

        # Unemployment: declining 9.8 → 7.8 with noise
        unemp = np.linspace(9.8, 7.8, n_dates) + rng.normal(0, 0.15, n_dates)

        for i, date in enumerate(DATES):
            is_holiday = int(date in HOLIDAY_DATES)

            # MarkDown only available post Nov 2011
            if date < pd.Timestamp("2011-11-01"):
                md1 = md2 = md3 = md4 = md5 = np.nan
            else:
                # Sparse markdowns — ~40% of rows have activity
                md1 = rng.choice([0, rng.uniform(500, 8000)], p=[0.6, 0.4])
                md2 = rng.choice([0, rng.uniform(200, 5000)], p=[0.7, 0.3])
                md3 = rng.choice([0, rng.uniform(100, 3000)], p=[0.75, 0.25])
                md4 = rng.choice([0, rng.uniform(50,  2000)], p=[0.8, 0.2])
                md5 = rng.choice([0, rng.uniform(500, 6000)], p=[0.65, 0.35])

            rows.append({
                "Store":       s,
                "Date":        date,
                "Temperature": round(temp_series[i], 2),
                "Fuel_Price":  round(fuel[i], 3),
                "MarkDown1":   None if np.isnan(md1) else round(md1, 2),
                "MarkDown2":   None if np.isnan(md2) else round(md2, 2),
                "MarkDown3":   None if np.isnan(md3) else round(md3, 2),
                "MarkDown4":   None if np.isnan(md4) else round(md4, 2),
                "MarkDown5":   None if np.isnan(md5) else round(md5, 2),
                "CPI":         round(cpi[i], 4),
                "Unemployment":round(unemp[i], 3),
                "IsHoliday":   is_holiday,
            })

    return pd.DataFrame(rows)

# ── Train ──────────────────────────────────────────────────────────────────────
DEPTS = list(range(1, 82))  # 81 departments

# Base weekly sales per department (heterogeneous)
DEPT_BASE = {d: rng.uniform(2000, 35000) for d in DEPTS}

# Seasonal multipliers by month (retail peaks in Nov/Dec)
SEASON = {1:0.85, 2:0.80, 3:0.88, 4:0.90, 5:0.92, 6:0.93,
          7:0.91, 8:0.93, 9:0.90, 10:0.95, 11:1.25, 12:1.45}

def make_train(stores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_dates = len(DATES)

    for _, store_row in stores_df.iterrows():
        s     = int(store_row["Store"])
        s_type = store_row["Type"]
        size   = store_row["Size"]
        size_mult = size / 150_000  # normalise around Type B

        for d in DEPTS:
            base = DEPT_BASE[d] * size_mult

            # Type multiplier
            type_mult = {"A": 1.15, "B": 1.0, "C": 0.75}[s_type]
            base *= type_mult

            # AR(1) process for correlated noise
            ar_noise = np.zeros(n_dates)
            ar_noise[0] = rng.normal(0, base * 0.05)
            for t in range(1, n_dates):
                ar_noise[t] = 0.7 * ar_noise[t-1] + rng.normal(0, base * 0.04)

            for i, date in enumerate(DATES):
                season_mult = SEASON[date.month]
                is_holiday  = int(date in HOLIDAY_DATES)
                holiday_mult = 1.0 + rng.uniform(0.15, 0.40) if is_holiday else 1.0

                weekly_sales = max(
                    0,
                    base * season_mult * holiday_mult + ar_noise[i]
                )

                rows.append({
                    "Store":        s,
                    "Dept":         d,
                    "Date":         date,
                    "Weekly_Sales": round(weekly_sales, 2),
                    "IsHoliday":    is_holiday,
                })

    return pd.DataFrame(rows)

# ── Test ───────────────────────────────────────────────────────────────────────
TEST_DATES = pd.date_range("2012-11-02", "2013-07-26", freq="W-FRI")

def make_test(stores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, store_row in stores_df.iterrows():
        s = int(store_row["Store"])
        for d in DEPTS:
            for date in TEST_DATES:
                is_holiday = int(date in HOLIDAY_DATES)
                rows.append({"Store": s, "Dept": d,
                              "Date": date, "IsHoliday": is_holiday})
    return pd.DataFrame(rows)

# ── Main ───────────────────────────────────────────────────────────────────────
def generate_all(force: bool = False):
    files = ["stores.csv", "train.csv", "features.csv", "test.csv"]
    existing = [f for f in files if os.path.exists(os.path.join(DATA_DIR, f))]

    if existing and not force:
        print(f"✅ Real data files found: {existing}. Skipping generation.")
        return

    print("⚙️  Generating synthetic Walmart-style data...")

    stores_df = make_stores()
    stores_df.to_csv(os.path.join(DATA_DIR, "stores.csv"), index=False)
    print(f"  stores.csv    — {len(stores_df):,} rows")

    features_df = make_features(stores_df)
    features_df.to_csv(os.path.join(DATA_DIR, "features.csv"), index=False)
    print(f"  features.csv  — {len(features_df):,} rows")

    train_df = make_train(stores_df)
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    print(f"  train.csv     — {len(train_df):,} rows")

    test_df = make_test(stores_df)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print(f"  test.csv      — {len(test_df):,} rows")

    print("✅ Synthetic data ready in data/")

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    generate_all(force=force)
