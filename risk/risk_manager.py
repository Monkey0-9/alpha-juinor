import pandas as pd
import numpy as np

TRADING_DAYS = 252

def volatility_targeting(
    returns: pd.Series,
    target_vol: float = 0.10,
    max_leverage: float = 1.5
) -> pd.Series:
    """
    Scale exposure based on realized volatility
    """
    realized_vol = returns.rolling(21).std() * np.sqrt(TRADING_DAYS)
    scale = target_vol / realized_vol
    scale = scale.clip(upper=max_leverage)
    return scale.fillna(0)

def drawdown_scaler(
    equity: pd.Series,
    max_dd_allowed: float = 0.10
) -> pd.Series:
    """
    Reduce exposure as drawdown increases
    """
    peak = equity.cummax()
    drawdown = 1 - equity / peak
    scaler = 1 - (drawdown / max_dd_allowed)
    scaler = scaler.clip(lower=0, upper=1)
    return scaler.fillna(0)
