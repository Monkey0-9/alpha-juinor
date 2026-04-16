
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger("RISK-GUARDIAN")

@dataclass
class RiskLimits:
    max_order_size_usd: float = 100000.0
    max_position_size_usd: float = 500000.0
    max_drawdown_pct: float = 0.02
    fat_finger_threshold_pct: float = 0.05 # Percent dev from last price

class RiskGuardian:
    """
    Ultra-fast Risk Guardian for pre-trade verification.
    Designed for <10μs latency in the execution path.
    """
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.current_exposure = 0.0
        self.daily_pnl = 0.0
        self.last_prices = {} # {symbol: price}

    def validate_order(self, 
                       symbol: str, 
                       quantity: float, 
                       price: float, 
                       side: str) -> (bool, str):
        """
        Pre-trade check. Returns (is_allowed, reason).
        """
        notional = abs(quantity * price)
        
        # 1. Fat Finger Check
        last_price = self.last_prices.get(symbol)
        if last_price:
            deviation = abs(price - last_price) / last_price
            if deviation > self.limits.fat_finger_threshold_pct:
                return False, f"FAT_FINGER: Price {price} deviates {deviation:.2%} from last {last_price}"

        # 2. Order Size Check
        if notional > self.limits.max_order_size_usd:
            return False, f"LIMIT_EXCEEDED: Order notional ${notional:,.2f} > max ${self.limits.max_order_size_usd:,.2f}"

        # 3. Position Size Check
        # Assume self.current_exposure is updated elsewhere
        if self.current_exposure + notional > self.limits.max_position_size_usd:
             return False, f"LIMIT_EXCEEDED: Resulting exposure would exceed max ${self.limits.max_position_size_usd:,.2f}"

        # 4. Drawdown Check
        if self.daily_pnl < -self.limits.max_drawdown_pct * 1000000: # Assuming $1M NAV
            return False, "HALTED: Daily drawdown limit reached"

        return True, "ALLOWED"

    def update_state(self, symbol: str, price: float, exposure_delta: float, pnl_delta: float):
        """Update risk state after execution or price tick."""
        self.last_prices[symbol] = price
        self.current_exposure += exposure_delta
        self.daily_pnl += pnl_delta
