import numpy as np
import pandas as pd
from typing import Dict

class RegimeDetector:
    """Detect market regimes using HMM or statistical thresholds"""
    
    def detect_regime(self, returns: pd.Series) -> str:
        """
        Identify Volatility and Trend regimes.
        """
        vol = returns.rolling(20).std() * np.sqrt(252)
        trend = returns.rolling(50).mean() * 252
        
        current_vol = vol.iloc[-1]
        current_trend = trend.iloc[-1]
        
        if current_vol > 0.25:
            if current_trend > 0:
                return "HIGH_VOL_BULL"
            else:
                return "HIGH_VOL_BEAR"
        else:
            if abs(current_trend) < 0.05:
                return "SIDEWAYS"
            return "STEADY_TREND"
