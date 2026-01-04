import pandas as pd
import numpy as np

def on_balance_volume(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV: cumulative volume adjusted by price direction
    """
    direction = np.sign(prices.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv

def volume_expansion(volume: pd.Series, window=20, threshold=1.5) -> pd.Series:
    """
    Detects abnormal participation
    """
    avg_volume = volume.rolling(window).mean()
    signal = (volume > avg_volume * threshold).astype(int)
    return signal.fillna(0)

def volatility_compression(prices: pd.Series, window=20, percentile=0.2) -> pd.Series:
    """
    Low volatility regimes often precede expansion
    """
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
    rolling_vol = returns.rolling(window).std()
    vol_rank = rolling_vol.rank(pct=True)
    signal = (vol_rank < percentile).astype(int)
    return signal.fillna(0)
