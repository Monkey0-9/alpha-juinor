import pandas as pd
import numpy as np

def market_regime_filter(prices: pd.Series, slow=200) -> pd.Series:
    """
    Block exposure in bear markets
    """
    sma = prices.rolling(slow).mean()
    allowed = (prices > sma).astype(int)
    return allowed.fillna(0)

def volatility_shock_filter(prices: pd.Series, window=20, shock_level=2.5) -> pd.Series:
    """
    Avoid trading during volatility explosions
    """
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
    rolling_vol = returns.rolling(window).std()
    shock = rolling_vol > rolling_vol.mean() * shock_level
    allowed = (~shock).astype(int)
    return allowed.fillna(0)

def liquidity_filter(volume: pd.Series, min_volume=5_000_000) -> pd.Series:
    """
    Avoid illiquid regimes
    """
    allowed = (volume > min_volume).astype(int)
    return allowed.fillna(0)
