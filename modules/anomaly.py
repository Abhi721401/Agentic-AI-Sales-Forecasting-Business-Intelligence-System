"""
modules/anomaly.py
──────────────────
Anomaly detection for Walmart weekly sales using:
  1. Isolation Forest  — ensemble method, effective for high-dimensional tabular data.
  2. Autoencoder       — reconstruction-error approach; anomalies have high error.

Both methods produce a binary flag and a continuous anomaly score.
Results are enriched with business context (holiday proximity, markdown spend).
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ANOMALY_FEATURES = [
    "Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Total_MarkDown", "IsHoliday", "Size", "Month", "Week",
    "Store_Mean_Sales", "Dept_Mean_Sales",
]


def _prepare(df: pd.DataFrame):
    available = [c for c in ANOMALY_FEATURES if c in df.columns]
    sub = df[available].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    return X, sub.index, scaler, available


# ── Isolation Forest ───────────────────────────────────────────────────────────

def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Isolation Forest: anomalies are points isolated by fewer splits.
    contamination=0.05 → top 5 % most isolated points flagged.
    Returns df with columns: IF_Anomaly (-1/1), IF_Score.
    """
    X, idx, scaler, _ = _prepare(df)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    preds  = iso.fit_predict(X)   # -1 = anomaly, 1 = normal
    scores = iso.score_samples(X) # lower = more anomalous

    result = df.loc[idx].copy()
    result["IF_Anomaly"] = preds
    result["IF_Score"]   = scores
    result["IF_Flag"]    = (preds == -1).astype(int)

    return result


# ── Autoencoder ────────────────────────────────────────────────────────────────

class _SimpleAutoencoder:
    """
    Lightweight NumPy autoencoder (no PyTorch/TF dependency).
    Architecture: input → bottleneck → reconstruction.
    Trained with SGD + MSE loss.
    """

    def __init__(self, input_dim: int, latent_dim: int = 4, lr: float = 0.01, epochs: int = 100):
        self.lr     = lr
        self.epochs = epochs
        rng = np.random.default_rng(42)

        # Encoder weights
        self.W1 = rng.normal(0, 0.1, (input_dim, latent_dim))
        self.b1 = np.zeros(latent_dim)

        # Decoder weights
        self.W2 = rng.normal(0, 0.1, (latent_dim, input_dim))
        self.b2 = np.zeros(input_dim)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(float)

    def _forward(self, X):
        Z1  = X @ self.W1 + self.b1
        A1  = self._relu(Z1)
        Z2  = A1 @ self.W2 + self.b2
        return Z1, A1, Z2  # Z2 = reconstruction

    def fit(self, X: np.ndarray):
        n = X.shape[0]
        for _ in range(self.epochs):
            Z1, A1, recon = self._forward(X)
            loss_grad = 2 * (recon - X) / n

            # Decoder gradients
            dW2 = A1.T @ loss_grad
            db2 = loss_grad.sum(axis=0)

            # Encoder gradients
            dA1 = loss_grad @ self.W2.T
            dZ1 = dA1 * self._relu_grad(Z1)
            dW1 = X.T @ dZ1
            db1 = dZ1.sum(axis=0)

            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        _, _, recon = self._forward(X)
        return np.mean((X - recon) ** 2, axis=1)


def run_autoencoder(df: pd.DataFrame, threshold_percentile: float = 95) -> pd.DataFrame:
    """
    Autoencoder anomaly detection:
    - Train on full dataset (unsupervised).
    - Compute per-row reconstruction error.
    - Flag rows whose error > threshold_percentile-th percentile.
    """
    X, idx, scaler, _ = _prepare(df)

    ae = _SimpleAutoencoder(input_dim=X.shape[1], latent_dim=4, lr=0.005, epochs=150)
    ae.fit(X)

    errors = ae.reconstruction_error(X)
    threshold = np.percentile(errors, threshold_percentile)

    result = df.loc[idx].copy()
    result["AE_Recon_Error"] = errors
    result["AE_Threshold"]   = threshold
    result["AE_Flag"]        = (errors > threshold).astype(int)

    return result


# ── Combined summary ───────────────────────────────────────────────────────────

def combined_anomaly_summary(if_df: pd.DataFrame, ae_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge both anomaly detection results.  Points flagged by BOTH methods
    are classified as 'High Confidence' anomalies.
    """
    needed = ["Store", "Dept", "Date", "Weekly_Sales",
              "IsHoliday", "Total_MarkDown", "IF_Flag", "IF_Score"]
    needed_ae = ["Date", "Store", "Dept", "AE_Flag", "AE_Recon_Error"]

    if_sub = if_df[[c for c in needed if c in if_df.columns]]
    ae_sub = ae_df[[c for c in needed_ae if c in ae_df.columns]]

    merged = if_sub.merge(ae_sub, on=["Date", "Store", "Dept"], how="inner")
    merged["Both_Flagged"] = ((merged["IF_Flag"] == 1) & (merged["AE_Flag"] == 1)).astype(int)
    merged["Confidence"] = merged["Both_Flagged"].map({1: "🔴 High", 0: "🟡 Single Method"})

    return merged.sort_values("AE_Recon_Error", ascending=False).reset_index(drop=True)


def explain_anomaly(row: pd.Series) -> str:
    """
    Generate a natural language explanation for a single anomaly row.
    """
    parts = []
    if row.get("IsHoliday", 0):
        parts.append("the week coincides with a holiday period")
    if row.get("Total_MarkDown", 0) > 5000:
        parts.append(f"significant promotional markdown spend (${row['Total_MarkDown']:,.0f})")
    if row.get("Weekly_Sales", 0) > 50000:
        parts.append("unusually high sales volume")
    elif row.get("Weekly_Sales", 0) < 1000:
        parts.append("abnormally low sales — possible store closure or data issue")

    if parts:
        return "Anomaly likely because: " + "; ".join(parts) + "."
    return "Anomaly detected by statistical model. Manual investigation recommended."
