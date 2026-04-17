import pandas as pd
from typing import List, Dict, Any, Callable
from datetime import datetime
from ..models.market import MarketBar
from .base import BaseAlpha, Signal

class WalkForwardValidator:
    """
    Utility for performing walk-forward validation of Alpha strategies.
    Ensures robustness by evaluating strategy performance across multiple rolling windows.
    """
    def __init__(self, window_size: int = 252, step_size: int = 63):
        self.window_size = window_size
        self.step_size = step_size

    def validate(self, strategy: BaseAlpha, data: List[MarketBar]) -> List[Signal]:
        """
        Executes a walk-forward simulation.
        The strategy is 'called' at every step, but only sees data up to that point.
        """
        signals = []
        n = len(data)
        
        # Start after the first window
        for i in range(self.window_size, n, self.step_size):
            # Isolation: strategy only sees data up to index i
            visible_data = data[:i]
            signal = strategy.generate_signal(visible_data)
            
            if signal:
                signals.append(signal)
                
        return signals

    def get_summary_stats(self, signals: List[Signal]) -> Dict[str, Any]:
        """
        Computes basic statistics on generated signals.
        """
        if not signals:
            return {"count": 0}
            
        values = [s.value for s in signals]
        return {
            "count": len(signals),
            "mean_signal": sum(values) / len(values),
            "positive_pct": len([v for v in values if v > 0]) / len(values)
        }
