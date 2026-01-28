
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpha_families.ml_alpha import MLAlpha
from data.processors.features import FeatureEngineer

@pytest.fixture
def mock_price_data():
    dates = pd.date_range("2023-01-01", periods=600, freq="B")
    df = pd.DataFrame(index=dates)
    # Generate random walk
    np.random.seed(42)
    df["Close"] = 100 * np.exp(np.random.normal(0.0005, 0.02, size=len(dates)).cumsum())
    df["Open"] = df["Close"] * (1 + np.random.normal(0, 0.005, size=len(dates)))
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    df["Volume"] = np.random.randint(1000, 10000, size=len(dates))
    return df

def test_feature_engineering(mock_price_data):
    """Test that feature engineering works correctly."""
    fe = FeatureEngineer()
    X = fe.compute_features(mock_price_data)

    assert not X.empty
    assert "rsi_14" in X.columns
    assert "ret_1d_lag_1" in X.columns

    # Check for non-NaN values (after dropna which happens in model, but here raw has NaNs at start)
    # Just check tail
    assert not X.iloc[-1].isnull().any()

def test_ml_alpha_training_and_prediction(mock_price_data):
    """Test MLAlpha model training and prediction."""
    # Skip if tensorflow not available
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow not available")

    fe = FeatureEngineer()
    model = MLAlpha(model_path=None)

    # Train
    model.train(mock_price_data)
    assert model.is_trained

    # Predict
    score = model.predict_conviction(mock_price_data)
    assert 0.0 <= score <= 1.0

