import pandas as pd
import numpy as np

TRADING_DAYS = 252

def risk_parity_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Simple inverse-volatility risk parity
    """
    vols = returns.std() * np.sqrt(TRADING_DAYS)
    inv_vol = 1 / vols
    weights = inv_vol / inv_vol.sum()
    return weights
