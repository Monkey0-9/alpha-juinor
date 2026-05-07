import pytest
import numpy as np
import pandas as pd
from nexus.math.risk import RiskEngine
from nexus.math.indicators import RegimeDetector

def test_risk_engine_var():
    engine = RiskEngine(confidence_level=0.99)
    returns = np.random.normal(0.001, 0.02, 1000)
    var = engine.calculate_var(returns)
    assert isinstance(var, float)
    assert var < 0 # VaR should be a loss (negative) for standard returns

def test_regime_detector():
    detector = RegimeDetector(window=10)
    # Simulate a bull market with enough data
    prices = np.linspace(100, 110, 30)
    df = pd.DataFrame({'close': prices})
    regime = detector.detect(df)
    assert regime in ["BULL", "SIDEWAYS", "TURBULENT"]

def test_risk_engine_cvar():
    engine = RiskEngine(confidence_level=0.99)
    returns = np.random.normal(0.001, 0.02, 1000)
    cvar = engine.calculate_cvar(returns)
    var = engine.calculate_var(returns)
    assert cvar <= var # CVaR is always worse than or equal to VaR
