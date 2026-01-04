
import pandas as pd
import numpy as np

def compute_rv(intra_returns: pd.Series) -> float:
    """
    RV = sum(r^2)
    """
    try:
        clean = intra_returns.replace([np.inf, -np.inf], np.nan).dropna()
        return float((clean**2).sum())
    except Exception:
        return 0.0

def compute_nvt(market_cap: float, daily_vol: float) -> float:
    """
    NVT = MarketCap / DailyTxnVolume
    """
    if daily_vol <= 0: return 0.0
    return market_cap / daily_vol
