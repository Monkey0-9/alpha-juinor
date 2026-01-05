"""
Simple Regime Detection

Lightweight regime detector based on volatility ratios and trend.
"""
import pandas as pd
import numpy as np
from typing import Tuple
from enum import Enum


class MarketRegime(Enum):
    """Market regime states."""
    LOW_VOL_BULL = "LOW_VOL_BULL"      # Low vol, uptrend
    HIGH_VOL_BULL = "HIGH_VOL_BULL"    # High vol, uptrend  
    LOW_VOL_BEAR = "LOW_VOL_BEAR"      # Low vol, downtrend
    HIGH_VOL_CRISIS = "HIGH_VOL_CRISIS"  # High vol, downtrend
    UNCERTAIN = "UNCERTAIN"


def detect_regime_simple(prices: pd.Series, 
                         short_window: int = 10,
                         long_window: int = 60,
                         vol_threshold: float = 1.5) -> Tuple[MarketRegime, dict]:
    """
    Simple regime detection using volatility ratio and trend.
    
    Args:
        prices: Price series
        short_window: Short volatility window (days)
        long_window: Long volatility window (days)
        vol_threshold: Ratio threshold for high vol regime
    
    Returns:
        Tuple of (regime, metadata)
    """
    if len(prices) < long_window:
        return MarketRegime.UNCERTAIN, {"reason": "insufficient_data"}
    
    # Calculate returns
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(returns) < long_window:
        return MarketRegime.UNCERTAIN, {"reason": "insufficient_returns"}
    
    # Volatility ratio
    short_vol = returns.tail(short_window).std() * np.sqrt(252)
    long_vol = returns.tail(long_window).std() * np.sqrt(252)
    
    if long_vol == 0:
        vol_ratio = 1.0
    else:
        vol_ratio = short_vol / long_vol
    
    is_high_vol = vol_ratio > vol_threshold
    
    # Trend detection (simple MA crossover)
    ma_short = prices.tail(short_window).mean()
    ma_long = prices.tail(long_window).mean()
    is_uptrend = ma_short > ma_long
    
    # Classify regime
    if is_uptrend and not is_high_vol:
        regime = MarketRegime.LOW_VOL_BULL
    elif is_uptrend and is_high_vol:
        regime = MarketRegime.HIGH_VOL_BULL
    elif not is_uptrend and not is_high_vol:
        regime = MarketRegime.LOW_VOL_BEAR
    else:  # not is_uptrend and is_high_vol
        regime = MarketRegime.HIGH_VOL_CRISIS
    
    metadata = {
        "vol_ratio": vol_ratio,
        "short_vol": short_vol,
        "long_vol": long_vol,
        "is_uptrend": is_uptrend,
        "is_high_vol": is_high_vol
    }
    
    return regime, metadata


def get_regime_risk_scalar(regime: MarketRegime) -> float:
    """
    Get risk scaling factor for regime.
    
    Returns:
        Scalar to apply to position sizing (0.0 to 1.0)
    """
    scalars = {
        MarketRegime.LOW_VOL_BULL: 1.0,      # Full risk
        MarketRegime.HIGH_VOL_BULL: 0.7,     # Moderate caution
        MarketRegime.LOW_VOL_BEAR: 0.5,      # Defensive
        MarketRegime.HIGH_VOL_CRISIS: 0.25,  # Extreme caution
        MarketRegime.UNCERTAIN: 0.6          # Conservative default
    }
    return scalars.get(regime, 0.5)
