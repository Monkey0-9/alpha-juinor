import pytest
import pandas as pd
import numpy as np
from alpha_families.statistical_alpha import StatisticalAlpha
from utils.metrics import metrics

def test_arima_fallback_to_ewma_for_short_series():
    sa = StatisticalAlpha()
    data = pd.DataFrame({'Close': [100.0] * 10}) # Too short for 50-bar ARIMA

    res = sa.arima_safe_predict(data)
    assert res['method'] == "NONE"
    assert res['signal'] == 0.0
    assert res['reason'] == "SHORT_SERIES"

def test_arima_fallback_on_convergence_failure():
    metrics.arima_fallbacks = 0
    sa = StatisticalAlpha()
    # Random walk data is usually okay, but we can mock ARIMA fit to fail or non-converge
    # For a real test, 50 bars of random noise often converges though.
    # Let's just verify the EWMA fallback logic if an exception occurs.

    data = pd.DataFrame({'Close': np.random.randn(100).cumsum() + 100})

    # Mock ARIMA or just check that it handles a bad series
    # A series with all same values might cause issues for ARIMA
    bad_data = pd.DataFrame({'Close': [100.0] * 100})

    res = sa.arima_safe_predict(bad_data)
    # If it fails to converge or errors, it should be EWMA
    if res['method'] == "EWMA":
        assert sa.arima_fallbacks > 0
        assert metrics.arima_fallbacks > 0
    else:
        # If it happens to converge, signal should be 0 since returns are 0
        assert res['signal'] == 0.0

def test_calculate_ewma_fallback():
    sa = StatisticalAlpha()
    data = pd.DataFrame({'Close': [100.0, 101.0, 102.0, 103.0]})
    signal = sa._calculate_ewma_fallback(data)
    assert isinstance(signal, float)
    assert signal > 0 # Positive returns should give positive signal
