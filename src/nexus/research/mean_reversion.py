import numpy as np
from typing import List, Optional
from .base import BaseAlpha, Signal
from ..models.market import MarketBar
from ..models.trade import OrderSide
from ..core.context import engine_context
from ..monitoring.profiler import profile_ns as profile_execution

class MeanReversionAlpha(BaseAlpha):
    """
    Standard Mean Reversion strategy using Z-scores.
    Signal = -(Price - MA) / StdDev
    """
    def __init__(self, name: str = "mean_reversion_20d", window: int = 20, threshold: float = 2.0):
        super().__init__(name)
        self.window = window
        self.threshold = threshold
        self.logger = engine_context.get_logger(f"alpha_{name}")

    @profile_execution("mean_reversion")
    def generate_signal(self, data: List[MarketBar]) -> Optional[Signal]:
        """
        Generates a mean reversion signal.
        """
        if len(data) < self.window:
            return None

        # Extract close prices
        closes = np.array([bar.close for bar in data[-self.window:]])
        current_price = closes[-1]
        
        ma = np.mean(closes)
        std = np.std(closes)
        
        if std == 0:
            return None
            
        z_score = (current_price - ma) / std
        
        # Signal is the negative of the Z-score (betting on reversion)
        signal_value = -z_score / self.threshold
        signal_value = max(-1.0, min(1.0, signal_value))
        
        # Threshold check
        if abs(z_score) < self.threshold * 0.5:
            return None # No strong signal
            
        side = OrderSide.BUY if signal_value > 0 else OrderSide.SELL
        
        return Signal(
            symbol=data[-1].symbol,
            timestamp=data[-1].timestamp,
            value=signal_value,
            side=side,
            confidence=min(1.0, abs(signal_value)),
            reason_code="MEAN_REVERSION_ZSCORE",
            metadata={"z_score": z_score, "ma": ma, "std": std}
        )
