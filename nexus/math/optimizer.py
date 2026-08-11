import numpy as np
import pandas as pd
from typing import Dict

class PortfolioOptimizer:
    """
    Handles position sizing and portfolio optimization.
    Implements a simple Kelly Criterion allocator and Equal Weighting.
    """
    def __init__(self, method='equal', max_exposure=1.0):
        self.method = method
        self.max_exposure = max_exposure

    def _equal_weighting(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Allocates equally among active signals."""
        active_assets = [k for k, v in signals.items() if abs(v) > 0.5]
        if not active_assets:
            return {k: 0.0 for k in signals}
            
        weight = self.max_exposure / len(active_assets)
        return {k: (weight if abs(signals[k]) > 0.5 else 0.0) for k in signals}
        
    def _kelly_weighting(self, signals: Dict[str, float], win_rates: Dict[str, float], payoff_ratios: Dict[str, float]) -> Dict[str, float]:
        """
        Fractional Kelly Criterion: f* = W - ((1 - W) / R)
        """
        weights = {}
        for symbol, signal in signals.items():
            if abs(signal) > 0.5 and symbol in win_rates and symbol in payoff_ratios:
                w = win_rates[symbol]
                r = payoff_ratios[symbol]
                if r > 0:
                    f = w - ((1 - w) / r)
                    # Use Half-Kelly for safety
                    weights[symbol] = max(0, f * 0.5) * self.max_exposure
                else:
                    weights[symbol] = 0.0
            else:
                weights[symbol] = 0.0
                
        # Normalize if exceeds max exposure
        total_weight = sum(weights.values())
        if total_weight > self.max_exposure:
            for k in weights:
                weights[k] = (weights[k] / total_weight) * self.max_exposure
                
        return weights

    def get_target_weights(self, signals: Dict[str, float], **kwargs) -> Dict[str, float]:
        if self.method == 'kelly':
            return self._kelly_weighting(signals, kwargs.get('win_rates', {}), kwargs.get('payoff_ratios', {}))
        return self._equal_weighting(signals)
