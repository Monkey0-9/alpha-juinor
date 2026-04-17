import pandas as pd
from typing import List, Optional
from .base import BaseAlpha, Signal
from ..models.market import MarketBar
from ..models.trade import OrderSide
from ..core.context import engine_context
from ..monitoring.profiler import profile_ns as profile_execution

class MomentumAlpha(BaseAlpha):
    """
    Classic cross-sectional momentum strategy.
    Signal = (Price_t / Price_{t-n}) - 1
    """
    def __init__(self, name: str = "momentum_12m", lookback: int = 252):
        super().__init__(name)
        self.lookback = lookback
        self.logger = engine_context.get_logger(f"alpha_{name}")

    @profile_execution("momentum_signal")
    def generate_signal(self, data: List[MarketBar]) -> Optional[Signal]:
        """
        Generates a momentum signal.
        Requires at least 'lookback' bars.
        """
        if len(data) < self.lookback:
            return None

        # Institutional systems avoid look-ahead bias by only looking at data[:now]
        # Here we assume 'data' is chronologically sorted up to 'now'.
        current_bar = data[-1]
        start_bar = data[-self.lookback]
        
        returns = (current_bar.close / start_bar.close) - 1
        
        # Normalize signal [-1, 1] using a simple threshold for demonstration
        # In a real system, this would be Z-scored across a universe.
        signal_value = max(-1.0, min(1.0, returns * 5.0)) 
        
        side = OrderSide.BUY if signal_value > 0 else OrderSide.SELL
        
        return Signal(
            symbol=current_bar.symbol,
            timestamp=current_bar.timestamp,
            value=signal_value,
            side=side,
            confidence=abs(signal_value),
            reason_code="MOMENTUM_EXTRAPOLATION",
            metadata={"lookback": self.lookback, "raw_return": returns}
        )
