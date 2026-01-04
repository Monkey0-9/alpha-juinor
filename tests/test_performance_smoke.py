
import pytest
import time
import pandas as pd
import numpy as np
from strategies.alpha import TrendAlpha
from data.processors.features import FeatureEngineer

def test_performance_smoke_features():
    """Ensure feature generation doesn't block for more than 2 seconds for small universe."""
    fe = FeatureEngineer()
    data = pd.DataFrame({
        "Open": np.random.randn(500).cumsum() + 100,
        "High": np.random.randn(500).cumsum() + 100,
        "Low": np.random.randn(500).cumsum() + 100,
        "Close": np.random.randn(500).cumsum() + 100,
        "Volume": np.random.randint(1000, 10000, 500)
    })
    
    t0 = time.time()
    fe.compute_features(data)
    duration = time.time() - t0
    assert duration < 2.0, f"Feature engineering too slow: {duration:.2f}s"

def test_performance_smoke_alpha():
    """Ensure alpha computation is efficient."""
    alpha = TrendAlpha()
    prices = pd.Series(np.random.randn(1000).cumsum() + 100)
    
    t0 = time.time()
    alpha.compute(prices)
    duration = time.time() - t0
    assert duration < 0.5, f"Alpha computation too slow: {duration:.2f}s"
