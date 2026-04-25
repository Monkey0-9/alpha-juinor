import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("RegimeDetector")

class SovereignRegimeDetector:
    """
    Detects market regimes using volatility and trend persistence (Hurst).
    Categorizes market into: BULL, BEAR, or TURBULENT.
    """
    def __init__(self, window=20):
        self.window = window

    def calculate_hurst(self, ts):
        """
        Calculates Hurst Exponent to determine trend persistence.
        H > 0.5: Trending
        H < 0.5: Mean-Reverting
        H = 0.5: Random Walk
        """
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]*2.0

    def detect(self, prices: pd.Series) -> str:
        if len(prices) < self.window:
            return "UNKNOWN"
        
        returns = prices.pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        trend = (prices.iloc[-1] / prices.iloc[-self.window]) - 1
        hurst = self.calculate_hurst(prices.values[-self.window:])

        if vol > 0.40:
            return "TURBULENT"
        elif trend > 0.02 and hurst > 0.5:
            return "BULL"
        elif trend < -0.02 and hurst > 0.5:
            return "BEAR"
        else:
            return "SIDEWAYS"
