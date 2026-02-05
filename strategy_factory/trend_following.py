import pandas as pd
import numpy as np
from typing import Optional
from .interface import StrategyInterface, Signal

class TrendFollowingStrategy(StrategyInterface):
    """
    12-Month Time-Series Momentum (TSMOm).
    Rationale: Captures long-term macro trends.
    Expected Correlation with MR: < 0 (Negative/Low).
    """

    def __init__(self, lookback_days: int = 252):
        self._name = "TrendFollowing_12M"
        self.lookback = lookback_days

    @property
    def name(self) -> str:
        return self._name

    def generate_signal(self, symbol: str, prices: pd.Series, regime_data: Optional[dict] = None) -> Signal:
        if len(prices) < self.lookback + 5:
            return Signal(symbol, 0.0, 0.0, False, {})

        # Calc 12-month return
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-self.lookback]

        momentum = (current_price - past_price) / past_price

        # Logic:
        # Positive Momentum -> Long
        # Negative Momentum -> Short (or Flat if constrained)

        strength = 0.0
        if momentum > 0:
            strength = 1.0
        else:
            strength = -1.0

        # Regime Adjustment
        regime_adjusted = False
        if regime_data:
            risk_mult = regime_data.get('risk_multiplier', 1.0)
            strength *= risk_mult
            regime_adjusted = True

        return Signal(
            symbol=symbol,
            strength=float(strength),
            confidence=min(abs(momentum) * 5, 1.0), # Higher momentum = higher confidence
            regime_adjusted=regime_adjusted,
            metadata={'momentum_12m': float(momentum)}
        )
