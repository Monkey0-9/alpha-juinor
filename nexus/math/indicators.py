import numpy as np
import pandas as pd
from typing import Dict

class RegimeDetector:
    """
    Market Regime Detection using volatility and trend analysis.
    """
    def __init__(self, window: int = 20):
        self.window = window

    def detect(self, data: pd.DataFrame) -> str:
        """
        Detects market regime: BULL, BEAR, SIDEWAYS, or TURBULENT.
        """
        if len(data) < self.window:
            return "SIDEWAYS"
        
        returns = data['close'].pct_change().dropna()
        vol = returns.rolling(window=self.window).std().iloc[-1]
        trend = (data['close'].iloc[-1] / data['close'].iloc[-self.window]) - 1
        
        if vol > 0.03:
            return "TURBULENT"
        if trend > 0.02:
            return "BULL"
        if trend < -0.02:
            return "BEAR"
        return "SIDEWAYS"

class HawkesProcess:
    """
    Self-exciting Hawkes Process for modeling volatility clustering.
    """
    def __init__(self, mu: float = 0.01, alpha: float = 0.1, beta: float = 0.5):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def calculate_intensity(self, events: np.ndarray) -> float:
        """
        Calculates intensity at the last event time.
        """
        if len(events) == 0:
            return self.mu
        
        t_last = events[-1]
        intensity = self.mu + np.sum(self.alpha * np.exp(-self.beta * (t_last - events[:-1])))
        return float(intensity)

def calculate_obi(bid_size: float, ask_size: float) -> float:
    """
    Calculates Order Book Imbalance (OBI).
    """
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    return (bid_size - ask_size) / total
