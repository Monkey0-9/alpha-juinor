
import pandas as pd
from typing import Dict, Optional

class RegimeModel:
    def __init__(self):
        self.state = "neutral"

    def infer(self, features: pd.DataFrame) -> str:
        """
        Return one of {'trend','range','high_vol'}; if unknown -> 'neutral'
        O(N) heuristic inference (Mocking Markov Model).
        """
        try:
            if features.empty: return "neutral"
            
            # Expecting 'Close'
            prices = features['Close']
            if len(prices) < 20: return "neutral"
            
            # PANDAS HYGIENE
            rets = prices.pct_change(fill_method=None).replace([float('inf'), float('-inf')], float('nan')).dropna()
            vol = rets.std()
            ma = prices.rolling(20).mean().iloc[-1]
            price = prices.iloc[-1]
            
            # Simple Heuristic States
            if vol > 0.02: # High Vol threshold
                return "high_vol"
            elif price > ma * 1.01:
                return "trend"
            elif price < ma * 0.99:
                return "trend"
            else:
                return "range"
                
        except Exception:
            return "neutral"
