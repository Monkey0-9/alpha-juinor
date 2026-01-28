"""
tests/test_feature_refresher.py
Unit tests for feature computation and refreshing
"""
import pytest
import pandas as pd
import numpy as np

from data.processors.features import compute_features_for_symbol
from features.contract import get_feature_list
from utils.errors import FeatureValidationError


def create_synthetic_ohlcv(n_rows=300):
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_rows, freq='D')

    # Generate realistic price data
    np.random.seed(42)
    close_prices = 100 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.02))

    df = pd.DataFrame({
        'Open': close_prices * (1 + np.random.randn(n_rows) * 0.01),
        'High': close_prices * (1 + np.abs(np.random.randn(n_rows)) * 0.02),
        'Low': close_prices * (1 - np.abs(np.random.randn(n_rows)) * 0.02),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, n_rows)
    }, index=dates)

    return df


def test_compute_features_contract_compliance():
    """Test that computed features match contract exactly."""
    df = create_synthetic_ohlcv(n_rows=300)

    features = compute_features_for_symbol(df, contract_name="ml_v1")

    # Check columns match contract
    expected_features = get_feature_list("ml_v1")
    assert list(features.columns) == expected_features

    # Check dtype
    assert all(dt == np.float32 for dt in features.dtypes)

    # Check n_features
    assert features.shape[1] == 28


def test_compute_features_deterministic():
    """Test that feature computation is deterministic."""
    df = create_synthetic_ohlcv(n_rows=300)

    features1 = compute_features_for_symbol(df, contract_name="ml_v1")
    features2 = compute_features_for_symbol(df.copy(), contract_name="ml_v1")

    # Should be identical
    pd.testing.assert_frame_equal(features1, features2)


def test_compute_features_no_nans_or_infs():
    """Test that computed features have no NaNs or infinities."""
    df = create_synthetic_ohlcv(n_rows=300)

    features = compute_features_for_symbol(df, contract_name="ml_v1")

    # Check no NaNs
    assert not features.isnull().any().any(), "Features contain NaN values"

    # Check no infinities
    assert not np.isinf(features.values).any(), "Features contain infinite values"


def test_compute_features_insufficient_data():
    """Test that insufficient data raises error."""
    df = create_synthetic_ohlcv(n_rows=100)  # Less than 252 required

    with pytest.raises(FeatureValidationError, match="Insufficient data"):
        compute_features_for_symbol(df, contract_name="ml_v1")


def test_compute_features_missing_columns():
    """Test that missing required columns raises error."""
    df = pd.DataFrame({
        'Close': np.random.randn(300),
        # Missing Open, High, Low, Volume
    })

    with pytest.raises(FeatureValidationError, match="Missing required price columns"):
        compute_features_for_symbol(df, contract_name="ml_v1")


def test_feature_ranges_reasonable():
    """Test that computed features have reasonable value ranges."""
    df = create_synthetic_ohlcv(n_rows=300)

    features = compute_features_for_symbol(df, contract_name="ml_v1")

    # RSI should be between 0 and 100
    assert (features["rsi_14"] >= 0).all() and (features["rsi_14"] <= 100).all()

    # Volatilities should be positive
    assert (features["vol_5d"] >= 0).all()
    assert (features["vol_20d"] >= 0).all()
    assert (features["vol_60d"] >= 0).all()

    # Regime flag should be 0 or 1
    assert features["regime_flag"].isin([0.0, 1.0]).all()


def test_feature_computation_with_missing_data():
    """Test handling of data with some missing values."""
    df = create_synthetic_ohlcv(n_rows=300)

    # Introduce some NaNs
    df.loc[df.index[50:52], 'Volume'] = np.nan

    # Should still compute features (fills with 0)
    features = compute_features_for_symbol(df, contract_name="ml_v1")

    assert features.shape[1] == 28
    assert not features.isnull().any().any()


def test_compute_features_shapes():
    """Test that output shape is correct for various input sizes."""
    for n_rows in [252, 300, 500, 1000]:
        df = create_synthetic_ohlcv(n_rows=n_rows)
        features = compute_features_for_symbol(df, contract_name="ml_v1")

        assert features.shape[1] == 28
        assert features.shape[0] == n_rows
