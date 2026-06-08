"""
setup.py
────────
One-command setup: installs dependencies, validates data, generates sample
data if Kaggle CSVs are absent, and runs a smoke-test of every module.

Usage:
    python setup.py            # full setup
    python setup.py --check    # only validate, don't install
    python setup.py --data     # only generate sample data
"""

import subprocess
import sys
import os
import importlib

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

REQUIRED_PACKAGES = [
    "streamlit", "pandas", "numpy", "scikit-learn",
    "statsmodels", "plotly", "scipy", "joblib",
    "matplotlib", "seaborn", "python-dotenv",
]

OPTIONAL_PACKAGES = {
    "shap":  "XAI/SHAP explanations",
    "groq":  "LLM Chatbot (Groq/LLaMA 3)",
    "fpdf2": "PDF Report Generation",
    "pyod":  "Advanced anomaly detection",
}


def banner(text: str, char: str = "─"):
    width = 60
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def install_packages():
    banner("📦 Installing required packages", "═")
    reqs = os.path.join(ROOT, "requirements.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", reqs, "-q"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌ Installation failed. Try: pip install -r requirements.txt")
        sys.exit(1)
    print("✅ All packages installed.")


def check_imports():
    banner("🔍 Checking imports")
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        mod = pkg.replace("-", "_").split("[")[0]
        if mod == "python_dotenv":
            mod = "dotenv"
        try:
            importlib.import_module(mod)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — NOT FOUND")
            all_ok = False

    print("\nOptional packages:")
    for pkg, desc in OPTIONAL_PACKAGES.items():
        try:
            importlib.import_module(pkg)
            print(f"  ✅ {pkg:10s} ({desc})")
        except ImportError:
            print(f"  ⚠️  {pkg:10s} ({desc}) — not installed; feature disabled")

    return all_ok


def check_data():
    banner("📂 Checking data files")
    files = ["train.csv", "test.csv", "stores.csv", "features.csv"]
    missing = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  ✅ {f:20s} ({size:.1f} KB)")
        else:
            print(f"  ⚠️  {f:20s} — not found")
            missing.append(f)

    if missing:
        print(f"\n  ℹ️  Missing: {missing}")
        print("  → Generating synthetic demo data...")
        gen_data()
    else:
        print("\n  ✅ All Kaggle data files present.")


def gen_data():
    banner("🏭 Generating synthetic data")
    gen_path = os.path.join(ROOT, "utils", "generate_sample_data.py")
    result = subprocess.run(
        [sys.executable, gen_path, "--force"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌ Data generation failed.")
        sys.exit(1)


def smoke_test():
    banner("🧪 Running smoke tests")
    tests = [
        ("Data loader",    "from modules.data_loader import load_and_clean_data"),
        ("Regression",     "from modules.regression import FEATURE_COLS"),
        ("Forecasting",    "from modules.forecasting import fit_arima"),
        ("Anomaly",        "from modules.anomaly import run_isolation_forest"),
        ("XAI",            "from modules.xai import shap_feature_importance"),
        ("Agent",          "from modules.agent import nl_to_pandas"),
        ("Charts",         "from utils.charts import LAYOUT"),
        ("Report",         "from modules.report_generator import RetailReport"),
    ]

    sys.path.insert(0, ROOT)
    all_pass = True
    for name, stmt in tests:
        try:
            exec(stmt)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_pass = False

    return all_pass


def check_env():
    banner("🔑 Environment variables")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        print(f"  ✅ GROQ_API_KEY set ({groq_key[:6]}...)")
    else:
        print("  ⚠️  GROQ_API_KEY not set — LLM chatbot disabled.")
        print("      Get a free key at: https://console.groq.com")
        print("      Then add to .env:  GROQ_API_KEY=gsk_...")


def print_launch():
    banner("🚀 Ready to launch!", "═")
    print("""
  Start the app:
      streamlit run app.py

  With Groq API key:
      GROQ_API_KEY=gsk_xxx streamlit run app.py

  Place real Kaggle data in:
      data/train.csv
      data/test.csv
      data/stores.csv
      data/features.csv

  Download from:
      https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
""")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--data" in args:
        gen_data()
    elif "--check" in args:
        check_imports()
        check_data()
        check_env()
        smoke_test()
    else:
        install_packages()
        ok_imports = check_imports()
        check_data()
        check_env()
        ok_smoke = smoke_test()
        print_launch()
        if not (ok_imports and ok_smoke):
            print("⚠️  Some checks failed. See above.")
            sys.exit(1)
