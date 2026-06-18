"""
test_preprocessing.py
----------------------
Unit tests for the preprocessing pipeline.

Covers
------
- Schema validation (pass and fail cases)
- Missing value imputation completeness
- Value-range validation
- Scaler output properties
- Pipeline end-to-end with synthetic data

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    FEATURE_COLS,
    handle_missing,
    remove_outliers_iqr,
    run_pipeline,
    scale_features,
    validate_schema,
    validate_value_ranges,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_sample_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Create a minimal valid DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id":        [f"C{i}" for i in range(n)],
        "recency_days":       rng.integers(1, 120, n),
        "frequency":          rng.integers(1, 20, n),
        "monetary_value":     rng.uniform(10, 1000, n),
        "session_count":      rng.integers(1, 60, n),
        "avg_session_min":    rng.uniform(1, 20, n),
        "categories_browsed": rng.integers(1, 10, n).astype(float),
        "discount_usage_pct": rng.uniform(0, 1, n),
        "mobile_pct":         rng.uniform(0, 1, n),
        "email_open_rate":    rng.uniform(0, 1, n),
        "cart_abandon_rate":  rng.uniform(0, 1, n),
        "true_cluster":       rng.integers(0, 4, n),
    })


# ── Schema Validation ──────────────────────────────────────────────────────────

class TestValidateSchema:
    def test_valid_schema_passes(self):
        df = make_sample_df()
        validate_schema(df)  # Should not raise

    def test_missing_column_raises(self):
        df = make_sample_df().drop(columns=["frequency"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_multiple_missing_columns_raises(self):
        df = make_sample_df().drop(columns=["frequency", "monetary_value"])
        with pytest.raises(ValueError):
            validate_schema(df)


# ── Value Range Validation ─────────────────────────────────────────────────────

class TestValidateValueRanges:
    def test_valid_data_unchanged(self):
        df = make_sample_df()
        result = validate_value_ranges(df)
        assert len(result) == len(df)

    def test_negative_recency_removed(self):
        df = make_sample_df()
        df.loc[0, "recency_days"] = -5
        result = validate_value_ranges(df)
        assert len(result) == len(df) - 1

    def test_out_of_range_discount_removed(self):
        df = make_sample_df()
        df.loc[0, "discount_usage_pct"] = 1.5  # > 1.0
        result = validate_value_ranges(df)
        assert len(result) == len(df) - 1

    def test_negative_monetary_removed(self):
        df = make_sample_df()
        df.loc[3, "monetary_value"] = -100
        result = validate_value_ranges(df)
        assert len(result) == len(df) - 1


# ── Missing Value Handling ─────────────────────────────────────────────────────

class TestHandleMissing:
    def test_no_nulls_after_imputation(self):
        df = make_sample_df()
        # Inject nulls
        df.loc[[0, 5, 10], "avg_session_min"] = np.nan
        df.loc[[2, 8], "email_open_rate"] = np.nan
        result = handle_missing(df)
        assert result[FEATURE_COLS].isnull().sum().sum() == 0

    def test_imputes_with_median(self):
        df = make_sample_df()
        expected_median = df["frequency"].median()
        df.loc[0, "frequency"] = np.nan
        result = handle_missing(df)
        assert result.loc[0, "frequency"] == pytest.approx(expected_median)

    def test_no_nulls_in_clean_data_unchanged(self):
        df = make_sample_df()
        result = handle_missing(df.copy())
        pd.testing.assert_frame_equal(df, result)


# ── Outlier Removal ────────────────────────────────────────────────────────────

class TestRemoveOutliersIQR:
    def test_removes_extreme_values(self):
        df = make_sample_df()
        # Inject an extreme outlier
        df.loc[0, "monetary_value"] = 999_999
        result = remove_outliers_iqr(df)
        assert 0 not in result.index or result.loc[result.index[0], "monetary_value"] < 999_999

    def test_result_smaller_than_input(self):
        df = make_sample_df()
        for i in range(5):
            df.loc[i, "session_count"] = 100_000
        result = remove_outliers_iqr(df)
        assert len(result) < len(df)

    def test_clean_data_mostly_retained(self):
        df = make_sample_df()
        result = remove_outliers_iqr(df)
        # With synthetic bounded data, >90% should be retained
        assert len(result) / len(df) > 0.90


# ── Scaler ─────────────────────────────────────────────────────────────────────

class TestScaleFeatures:
    def test_output_shape(self):
        df = make_sample_df()
        X, scaler = scale_features(df)
        assert X.shape == (len(df), len(FEATURE_COLS))

    def test_zero_mean_unit_variance(self):
        df = make_sample_df(n=500)
        X, _ = scale_features(df)
        assert np.abs(X.mean(axis=0)).max() < 0.01
        assert np.abs(X.std(axis=0) - 1.0).max() < 0.01

    def test_scaler_inverse_transform(self):
        df = make_sample_df()
        X, scaler = scale_features(df)
        X_back = scaler.inverse_transform(X)
        np.testing.assert_allclose(X_back, df[FEATURE_COLS].values, atol=1e-6)


# ── End-to-End Pipeline ────────────────────────────────────────────────────────

class TestRunPipeline:
    def test_pipeline_returns_correct_shapes(self, tmp_path):
        # Write a small synthetic CSV
        df = make_sample_df(n=100)
        csv_path = tmp_path / "test_customers.csv"
        df.to_csv(csv_path, index=False)

        X, df_clean, scaler = run_pipeline(csv_path)

        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[1] == len(FEATURE_COLS)
        assert len(df_clean) <= 100  # may be fewer after outlier removal

    def test_pipeline_no_nulls_in_output(self, tmp_path):
        df = make_sample_df(n=150)
        df.loc[[0, 5], "avg_session_min"] = np.nan
        csv_path = tmp_path / "test_customers.csv"
        df.to_csv(csv_path, index=False)

        X, df_clean, _ = run_pipeline(csv_path)
        assert np.isnan(X).sum() == 0

    def test_pipeline_saves_output(self, tmp_path):
        df = make_sample_df(n=100)
        csv_path = tmp_path / "test_customers.csv"
        out_path = tmp_path / "output.csv"
        df.to_csv(csv_path, index=False)

        run_pipeline(csv_path, output_path=out_path)
        assert out_path.exists()
