"""
preprocessing.py
----------------
Data preprocessing and validation pipeline for customer segmentation.

Steps
-----
1. Load raw CSV
2. Validate schema and data types
3. Handle missing values
4. Remove outliers (IQR method)
5. Encode / scale features
6. Save processed dataset

Usage
-----
    python src/preprocessing.py
    # or import and call:
    from src.preprocessing import run_pipeline
    X_scaled, df_clean = run_pipeline("data/processed/customers.csv")
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary_value",
    "session_count",
    "avg_session_min",
    "categories_browsed",
    "discount_usage_pct",
    "mobile_pct",
    "email_open_rate",
    "cart_abandon_rate",
]

EXPECTED_SCHEMA = {
    "customer_id": "object",
    "recency_days": "int64",
    "frequency": "int64",
    "monetary_value": "float64",
    "session_count": "int64",
    "avg_session_min": "float64",
    "categories_browsed": "float64",  # may have NaN → stored as float
    "discount_usage_pct": "float64",
    "mobile_pct": "float64",
    "email_open_rate": "float64",
    "cart_abandon_rate": "float64",
}

# ── Validation ─────────────────────────────────────────────────────────────────


def validate_schema(df: pd.DataFrame) -> None:
    """Assert that required columns exist and raise informative errors."""
    missing = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log.info("✅ Schema validation passed (%d rows, %d cols)", *df.shape)


def validate_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and remove rows with logically impossible values."""
    initial = len(df)
    df = df[df["recency_days"] >= 0]
    df = df[df["frequency"] >= 0]
    df = df[df["monetary_value"] >= 0]
    df = df[df["discount_usage_pct"].between(0, 1)]
    df = df[df["mobile_pct"].between(0, 1)]
    df = df[df["email_open_rate"].between(0, 1)]
    df = df[df["cart_abandon_rate"].between(0, 1)]
    removed = initial - len(df)
    if removed:
        log.warning("⚠️  Removed %d rows with out-of-range values", removed)
    else:
        log.info("✅ Value-range validation passed (no invalid rows)")
    return df


# ── Missing Values ─────────────────────────────────────────────────────────────


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation strategy:
    - Continuous features → median imputation (robust to skew)
    - Report missing rates before and after
    """
    missing_before = df[FEATURE_COLS].isnull().sum()
    log.info("Missing values before imputation:\n%s", missing_before[missing_before > 0].to_string())

    for col in FEATURE_COLS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info("  Imputed '%s' with median=%.3f", col, median_val)

    assert df[FEATURE_COLS].isnull().sum().sum() == 0, "Imputation incomplete!"
    log.info("✅ Missing value imputation complete")
    return df


# ── Outlier Removal ────────────────────────────────────────────────────────────


def remove_outliers_iqr(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """
    Remove rows where any feature is beyond factor * IQR from Q1/Q3.
    Uses a conservative factor=3.0 to only remove extreme outliers.
    """
    initial = len(df)
    mask = pd.Series([True] * len(df), index=df.index)
    for col in FEATURE_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        col_mask = df[col].between(lower, upper)
        mask &= col_mask
    df_clean = df[mask].reset_index(drop=True)
    removed = initial - len(df_clean)
    log.info("✅ Outlier removal: dropped %d rows (%.1f%%)", removed, 100 * removed / initial)
    return df_clean


# ── Scaling ────────────────────────────────────────────────────────────────────


def scale_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Standardize features to zero mean and unit variance.
    Returns the scaled array and the fitted scaler (for inverse transform).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS])
    log.info("✅ Features standardized (mean≈0, std≈1)")
    return X, scaler


# ── Pipeline ───────────────────────────────────────────────────────────────────


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    input_path  : Path to raw customers.csv
    output_path : Optional path to save cleaned DataFrame

    Returns
    -------
    X_scaled  : np.ndarray of shape (n_samples, n_features) — model-ready
    df_clean  : Cleaned DataFrame with original columns
    scaler    : Fitted StandardScaler for inverse transforms
    """
    log.info("── Loading data from %s", input_path)
    df = pd.read_csv(input_path)

    validate_schema(df)
    df = validate_value_ranges(df)
    df = handle_missing(df)
    df = remove_outliers_iqr(df)
    X_scaled, scaler = scale_features(df)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_out = df.copy()
        df_out[FEATURE_COLS] = X_scaled
        df_out.to_csv(out, index=False)
        log.info("✅ Processed dataset saved → %s", out)

    log.info("── Pipeline complete. Final shape: %s", df.shape)
    return X_scaled, df, scaler


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_csv = Path("data/processed/customers.csv")
    output_csv = Path("data/processed/customers_scaled.csv")

    if not input_csv.exists():
        log.error("Dataset not found. Run `python data/generate_synthetic.py` first.")
        sys.exit(1)

    X, df_clean, scaler = run_pipeline(input_csv, output_csv)
    print(f"\nReady for modeling: X shape = {X.shape}")
