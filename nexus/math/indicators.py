from typing import Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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
        
        try:
            # Robust selection: take the first 'close' column if multiple exist
            close_col = data['close']
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            
            returns = close_col.pct_change().dropna()
            vol_series = returns.rolling(window=self.window).std()
            if vol_series.empty:
                return "SIDEWAYS"
            vol = float(vol_series.iloc[-1])
            
            c_last = float(close_col.iloc[-1])
            c_prev = float(close_col.iloc[-self.window])
            trend = (c_last / c_prev) - 1
            
            if vol > 0.03:
                return "TURBULENT"
            if trend > 0.02:
                return "BULL"
            if trend < -0.02:
                return "BEAR"
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return "UNKNOWN"
        return "SIDEWAYS"


class HawkesProcess:
    """
    Self-exciting Hawkes Process for modeling volatility clustering.
    """
    def __init__(self, mu: float = 0.01, alpha: float = 0.1, beta: float = 0.5):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta

    def calculate_intensity(self, events: np.ndarray[Any, Any]) -> float:
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
