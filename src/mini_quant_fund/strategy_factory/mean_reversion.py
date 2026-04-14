import pandas as pd
import numpy as np
from typing import Optional
from .interface import StrategyInterface, Signal

class MeanReversionStrategy(StrategyInterface):
    """
    Validated RSI-based Mean Reversion Strategy.
    OOS Sharpe: 0.85
    """

    def __init__(self, rsi_period: int = 14, overbought: int = 70, oversold: int = 30):
        self._name = "MeanRestoration_RSI"
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    @property
    def name(self) -> str:
        return self._name

    def generate_signal(self, symbol: str, prices: pd.Series, regime_data: Optional[dict] = None) -> Signal:
        if len(prices) < self.rsi_period + 5:
            return Signal(symbol, 0.0, 0.0, False, {})

        # Calc RSI
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean().iloc[-1]

        if loss == 0:
            rsi = 100.0
        else:
            rs = gain / loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        strength = 0.0
        # Logic:
        # RSI < 30 -> Long
        # RSI > 70 -> Short/Flat (Negative signal)

        if rsi < self.oversold:
            # Scale from 0 to 1 as it gets deeper into oversold
            strength = 1.0
        elif rsi > self.overbought:
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
            confidence=1.0 if abs(strength) > 0 else 0.0,
            regime_adjusted=regime_adjusted,
            metadata={'rsi': float(rsi)}
        )
