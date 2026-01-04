
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.ml_models.ml_alpha import MLAlpha
from data.processors.features import FeatureEngineer
from portfolio.optimizer import MeanVarianceOptimizer

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
    fe = FeatureEngineer()
    X = fe.compute_features(mock_price_data)
    
    assert not X.empty
    assert "rsi_14" in X.columns
    assert "ret_1d_lag_1" in X.columns
    
    # Check for non-NaN values (after dropna which happens in model, but here raw has NaNs at start)
    # Just check tail
    assert not X.iloc[-1].isnull().any()

def test_ml_alpha_training_and_prediction(mock_price_data):
    fe = FeatureEngineer()
    model = MLAlpha(fe)
    
    # Train
    model.train(mock_price_data)
    assert model.is_trained
    
    # Predict
    score = model.predict_conviction(mock_price_data)
    assert 0.0 <= score <= 1.0
    
    # Check variation
    # Mock data is random walk, prediction might be close to 0.5 or not, but should run

def test_optimizer():
    optimizer = MeanVarianceOptimizer(max_weight=0.5)
    
    # Mock efficient frontier components
    tickers = ["A", "B", "C"]
    
    # 0.5% return, 1% return, 0.2% return
    er = pd.Series([0.005, 0.01, 0.002], index=tickers)
    
    # Simple diagonal covariance
    cov = pd.DataFrame(
        [[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.02]],
        index=tickers, columns=tickers
    )
    
    weights = optimizer.optimize(er, cov)
    
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 1e-4
    
    # Asset B has highest return but high risk. Asset C has low return low risk.
    # Optimizer should find tradeoff.
    print(weights)
