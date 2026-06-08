# 🛒 RetailGPT — Agentic AI Sales Forecasting & Business Intelligence System

> **MSc-level portfolio project** | Walmart Sales Forecasting | Streamlit + Python

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---
# Short Demo Video

![RetailGPT Demo](https://github.com/user-attachments/assets/6d437d4d-e537-4f71-a0ae-4c01ca5f4a18)

## 🎥 Demo Video

[![Watch Demo](https://img.shields.io/badge/▶%20Watch%20Demo-RetailGPT-red)](https://drive.google.com/file/d/17abSuwBPnby2LGjB8gD0wscXtD8ji9Ad/view?usp=drivesdk)

# What You’ll See

* End-to-end Walmart sales forecasting workflow
* Interactive KPI dashboard
* Regression analysis (OLS, Ridge, Lasso)
* ARIMA & SARIMA forecasting
* Anomaly detection using Isolation Forest
* Explainable AI with SHAP and PDP
* Agentic AI business assistant
* Automated PDF report generation

👉 Click the badge above to watch the full demonstration.


## 📐 Architecture Overview

```
retailgpt/
├── app.py                          ← Streamlit entry point + navigation
├── data/                           ← Place Kaggle CSVs here
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   └── features.csv
├── modules/                        ← Core analytical modules
│   ├── data_loader.py              ← Data pipeline: load → clean → feature engineer
│   ├── regression.py               ← OLS, Ridge, Lasso + VIF + coefficient analysis
│   ├── forecasting.py              ← ARIMA, SARIMA + ADF + ACF/PACF
│   ├── anomaly.py                  ← Isolation Forest + Autoencoder (NumPy)
│   ├── xai.py                      ← SHAP + PDP + business narratives
│   ├── agent.py                    ← Agentic AI + RAG + NL→Pandas + Groq chatbot
│   └── report_generator.py         ← Automated PDF report (fpdf2)
├── pages/                          ← Streamlit page renderers
│   ├── home.py                     ← KPI dashboard + trend charts
│   ├── data_exploration.py         ← EDA: distributions, correlations, outliers
│   ├── regression.py               ← Regression analysis UI
│   ├── forecasting.py              ← ARIMA/SARIMA forecasting UI
│   ├── anomaly.py                  ← Anomaly detection UI
│   ├── xai.py                      ← Explainability UI (SHAP, PDP, narratives)
│   ├── agent.py                    ← Agentic chatbot UI
│   └── report.py                   ← PDF report generation UI
├── utils/
│   ├── charts.py                   ← Plotly chart factory (dark theme)
│   ├── helpers.py                  ← Formatting, session state, UI components
│   └── generate_sample_data.py     ← Synthetic data generator for demo
├── .streamlit/config.toml          ← Dark theme config
├── .env.example                    ← API key template
├── requirements.txt
└── setup.py                        ← One-command setup + validation
```

---

## 🚀 Quick Start

### Option A — With real Kaggle data (recommended)

```bash
# 1. Clone / download this project
cd retailgpt

# 2. Download data from Kaggle
# https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
# Place: train.csv, test.csv, stores.csv, features.csv → into data/

# 3. Run setup
python setup.py

# 4. (Optional) Add Groq API key for chatbot
cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_your_key

# 5. Launch
streamlit run app.py
```

### Option B — Demo without Kaggle data

```bash
# Synthetic data is auto-generated on first launch
python setup.py
streamlit run app.py
```

---

## 📊 Features by Page

### 🏠 Home Dashboard
- 6-metric KPI strip (revenue, avg sales, peak, stores, depts, holiday uplift)
- Overall weekly sales trend with holiday markers
- Top 15 stores bar chart
- Monthly revenue by year (grouped bar)
- Store type pie chart
- Holiday vs non-holiday box plot
- Top 10 departments
- Store size vs total revenue scatter
- Named holiday uplift comparison
- Data cleaning decision log

### 🔍 Data Exploration
- Sales distribution (linear + log scale) with marginal boxplots
- Missing value heatmap after cleaning
- Descriptive statistics table
- Per-store multi-line trend chart
- Department × Month heatmap
- Correlation matrix (colour-coded)
- MarkDown vs Sales scatter
- Unemployment vs Sales scatter
- Outlier root cause classification (IQR + Z-score)
- Interactive store / dept / year filters

### 📈 Regression Analysis
- Three models: **OLS**, **Ridge** (α=10), **Lasso** (α=1)
- 80/20 chronological train/test split (no temporal leakage)
- Metrics: R², Adjusted R², RMSE, MAE, MAPE
- Colour-coded model comparison table
- Actual vs Predicted scatter
- Residuals histogram + residuals-over-time scatter
- Coefficient bar chart (signed, colour-coded)
- Full statsmodels OLS summary (p-values, t-stats, CI)
- VIF chart with severity flags (OK / Moderate / High)
- Manual prediction widget

### 📅 Forecasting
- Store-level or Store+Department granularity
- ADF stationarity test with metric display
- ACF/PACF plots with 95% Bartlett bounds
- ARIMA: AIC-optimised grid search over p,q ∈ {0,1,2}
- SARIMA: seasonal order search, m=52 (annual weekly cycle)
- Forecast charts with 95% CI shading
- RMSE, MAE, MAPE for each model
- Model comparison table + AIC-based winner declaration
- 4 / 12 / 52-week horizons
- Multi-store overlay comparison chart
- Forecast values table

### 🚨 Anomaly Detection
- **Isolation Forest**: contamination % slider (1–15%)
- **Autoencoder** (NumPy, no PyTorch): reconstruction error histogram + threshold line
- Combined view: dual-flagged = High Confidence anomalies
- Per-anomaly natural language explanations
- Anomaly confidence breakdown pie chart

### 🧠 Explainable AI
- SHAP global feature importance bar chart
- SHAP values heatmap (top 10 features × samples)
- SHAP waterfall chart (local explanation per observation)
- Partial Dependence Plots for any continuous feature
- Business narrative (markdown for non-technical stakeholders)
- Feature selection advice for Lasso

### 🤖 Agentic AI Assistant
- **6 tools**: sales query, store summary, store comparison, trend analysis, anomaly report, forecast summary
- ReAct routing: intent → tool → grounded LLM response
- RAG: dataset statistics + tool output injected into every LLM call
- NL → Pandas converter (offline, no API key needed)
- Multi-turn conversation with memory (last 10 turns)
- Example query buttons in sidebar
- RAG context inspector tab

### 📄 Report Generation
- Configurable PDF report (fpdf2)
- Sections: cover, KPIs, dataset overview, regression, forecast, anomalies, recommendations, methodology
- Auto-generated or custom recommendations
- Download button

---

## 🔑 API Key Setup (Groq / LLaMA 3)

1. Sign up at [console.groq.com](https://console.groq.com) (free)
2. Create an API key
3. Either:
   - Add to `.env`: `GROQ_API_KEY=gsk_...`
   - Or paste into the sidebar input on the Agentic AI page

Without a key, all features except LLM chat responses still work.

---

## 📐 Modeling Decisions

### Data Cleaning
| Feature | Decision | Justification |
|---|---|---|
| MarkDown1–5 NaN | Replace with 0 | NaN concentrated before Nov 2011 — absence of record = no promotion |
| Temperature NaN | Store-wise median | Robust to seasonal extremes; sparse missingness |
| Fuel_Price NaN | Store-wise median | Same rationale; store-level regional pattern |
| CPI NaN | Time-aware linear interpolation within store | Economic indicators evolve smoothly; interpolation respects continuity |
| Unemployment NaN | Time-aware linear interpolation within store | Same rationale |
| Outliers | Flagged, not removed | Most correspond to legitimate holiday demand spikes |

### Regression
| Decision | Reasoning |
|---|---|
| 80/20 chronological split | Retail data has temporal dependencies; random split would leak future via lag features |
| Ridge preferred | MarkDown features are correlated → high VIF → L2 shrinkage reduces variance |
| Standardisation | Allows coefficient magnitudes to be compared across different-scale features |
| VIF threshold = 10 | Standard econometric convention; VIF > 10 indicates problematic multicollinearity |

### Time Series
| Decision | Reasoning |
|---|---|
| ADF test before ARIMA | Must verify stationarity assumption; d determined by data, not assumed |
| AIC for order selection | Balances fit vs complexity; prevents overfitting on 143-week series |
| m=52 for SARIMA | Weekly data with annual cycle (Christmas/Thanksgiving repeat every 52 weeks) |
| 95% CI | Standard prediction interval; captures model parameter uncertainty |

### Anomaly Detection
| Decision | Reasoning |
|---|---|
| contamination=5% | Conservative estimate; retail has real event spikes not noise |
| Autoencoder p95 threshold | Flags the most extreme reconstruction failures while tolerating normal variance |
| Dual flagging | Points detected by both methods are more likely true anomalies |

---

## 📦 Dependencies

```
streamlit>=1.32.0       # UI framework
pandas>=2.0.0           # Data manipulation
numpy>=1.26.0           # Numerical computing
scikit-learn>=1.4.0     # ML models, preprocessing
statsmodels>=0.14.0     # ARIMA, SARIMA, OLS, ADF
plotly>=5.20.0          # Interactive charts
shap>=0.45.0            # Explainable AI
groq>=0.5.0             # LLM API client
fpdf2>=2.7.9            # PDF generation
scipy>=1.12.0           # Statistics (Z-score, IQR)
python-dotenv>=1.0.0    # .env file loading
```

---

## 🎓 Academic References

1. **SARIMA for retail forecasting**: Cheema & Bhatnagar (2021). SARIMA outperforms ARIMA and exponential smoothing for seasonal retail demand.
2. **SHAP values**: Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
3. **Isolation Forest**: Liu et al. (2008). *Isolation Forest*. ICDM.
4. **Variance Inflation Factor**: Kutner et al. (2005). *Applied Linear Statistical Models*, 5th ed.
5. **Ridge Regression**: Hoerl & Kennard (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems.
6. **AIC model selection**: Akaike (1974). A New Look at the Statistical Model Identification.

---

## 📝 License

MIT License — free for academic and commercial use.
