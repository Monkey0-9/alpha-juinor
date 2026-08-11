import os
import sys
import numpy as np
import pandas as pd

# Ensure our local extensions are loadable
mingw_bin = os.path.expanduser(r"~\scoop\apps\mingw\current\bin")
if os.path.exists(mingw_bin):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(mingw_bin)
    else:
        os.environ["PATH"] = mingw_bin + os.pathsep + os.environ["PATH"]

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "nexus", "cpp_extensions"
        )
    ),
)

# Test indicators
from nexus.math.indicators import (  # noqa: E402
    compute_hurst_exponent,
    HawkesProcess,
)
from nexus.core.alpha import AlphaEngine  # noqa: E402
from nexus.math.optimization import PortfolioOptimizer  # noqa: E402


def test_hurst_exponent():
    prices = pd.Series(np.linspace(10, 20, 100))  # Trending
    hurst = compute_hurst_exponent(prices)
    assert hurst > 0.5


def test_hawkes_process():
    hawkes = HawkesProcess(mu=0.01, alpha=0.1, beta=0.5)
    returns = np.random.normal(0, 0.01, 100)
    intensity = hawkes.calculate_intensity(returns)
    assert intensity >= 0.01


def test_alpha_engine_entropy():
    engine = AlphaEngine()
    returns = np.random.normal(0, 0.01, 100)
    entropy_filter = engine._compute_entropy_filter(returns)
    assert 0.0 <= entropy_filter <= 1.0


def test_alpha_engine_vwap():
    engine = AlphaEngine()
    df = pd.DataFrame(
        {
            "close": np.random.uniform(10, 20, 100),
            "high": np.random.uniform(15, 25, 100),
            "low": np.random.uniform(5, 15, 100),
            "volume": np.random.uniform(100, 1000, 100),
        }
    )
    signal = engine._compute_vwap_signal(df, 15.0)
    assert -1.0 <= signal <= 1.0


def test_portfolio_optimizer():
    opt = PortfolioOptimizer()
    symbols = ["AAPL", "MSFT"]
    signals = [0.9, -0.9]

    # Needs historical data to trigger correlation penalty
    df_aapl = pd.DataFrame({"close": np.linspace(100, 150, 50)})
    df_msft = pd.DataFrame(
        {"close": np.linspace(200, 300, 50)}
    )  # Highly correlated
    hist_data = {"AAPL": df_aapl, "MSFT": df_msft}

    weights = opt.optimize_weights(symbols, signals, hist_data)

    assert "AAPL" in weights
    assert "MSFT" in weights
    assert sum(weights.values()) <= 1.0
