
import pandas as pd
import numpy as np

class AuctionConfidence:
    """
    Auction Market Theory Confidence Scorer.
    
    Logic:
    - Measures 'Structure Quality' via VWAP and Value Area relationships.
    - If price is extending AWAY from VWAP with Volume -> TRENDING (High Confidence).
    - If price is chopping AROUND VWAP with low Volume -> BALANCED (Low Trend Confidence).
    """
    
    def __init__(self, window: int = 20):
        self.window = window

    def compute_confidence(self, close: pd.Series, volume: pd.Series, high: pd.Series, low: pd.Series) -> float:
        """
        Returns a scalar 0.0 to 1.0 representing confidence in the current directional move.
        1.0 = Strong Imbalance (Trend)
        0.0 = Perfect Balance (Chop)
        """
        try:
            if len(close) < self.window:
                return 1.0 # Neutral/Passthrough

            # Calculate Anchored VWAP (Window)
            # VWAP = Cumulative(Price * Vol) / Cumulative(Vol)
            # Here we do Rolling VWAP for safety
            
            pv = (close * volume).rolling(self.window).sum()
            v_sum = volume.rolling(self.window).sum()
            
            # Avoid div/0
            vwap = pv / v_sum.replace(0, 1) 
            
            current_price = close.iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            # Distance from VWAP normalized by ATR (or StdDev)
            std_dev = close.rolling(self.window).std().iloc[-1]
            if std_dev == 0: std_dev = close.iloc[-1] * 0.01
            
            dist_z = abs(current_price - current_vwap) / std_dev
            
            # Map Z-score to Confidence
            # Z < 0.5 (Inside noise) -> Confidence ~0.5
            # Z > 2.0 (Trend) -> Confidence ~1.0
            
            confidence = np.clip(0.5 + (dist_z - 0.5) * 0.25, 0.2, 1.0)
            
            return float(confidence)

        except Exception:
            return 1.0
