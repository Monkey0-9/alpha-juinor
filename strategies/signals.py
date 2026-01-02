import pandas as pd
import numpy as np

def sma_trend_signal(prices: pd.Series, fast=50, slow=200) -> pd.Series:
    """
    Long-only trend regime.
    1 = risk-on
    0 = risk-off
    """
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    signal = (sma_fast > sma_slow).astype(int)
    return signal.fillna(0)

def momentum_signal(prices: pd.Series, lookback=252, skip_recent=21) -> pd.Series:
    """
    12-1 momentum (institutional standard)
    """
    past_return = prices.shift(skip_recent) / prices.shift(lookback) - 1
    signal = (past_return > 0).astype(int)
    return signal.fillna(0)
