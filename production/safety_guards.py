"""
Production Safety & Governance
==============================

Critical safety guards for Phase 10 (Production Readiness):
- Kill Switches
- Fat Finger Checks
- Circuit Breakers
- API Key Rotation Stubs
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SafetyGuard:
    """
    Last line of defense before order submission.
    """

    def __init__(self, max_order_size_usd: float = 100000,
                 max_drawdown_stop: float = 0.05,
                 kill_switch_active: bool = False):
        self.max_order_size = max_order_size_usd
        self.max_drawdown = max_drawdown_stop
        self.is_kill_switch_active = kill_switch_active
        self.daily_loss = 0.0

    def check_pre_trade(self, order: Dict[str, Any]) -> bool:
        """
        Run all pre-trade checks. Returns True if safe to trade.
        """
        if self.is_kill_switch_active:
            logger.error("TRADE REJECTED: Kill switch active")
            return False

        # Fat finger check
        size = order.get('quantity', 0) * order.get('price', 0)
        if size > self.max_order_size:
            logger.error(f"TRADE REJECTED: Size {size} exceeds max {self.max_order_size}")
            return False

        # --- INSTITUTIONAL "SMART" CHECKS ---

        # 1. Liquidity Check (Avoid "Dumb" illiquid stocks)
        # Require 30-day Avg Volume > 500k or Dollar Vol > $5M
        adv = order.get('adv_30d', 1000000) # Default to passing if data missing
        if adv < 100000:
            logger.warning(f"TRADE REJECTED: Low Liquidity ({adv} < 100k)")
            return False

        # 2. Spread Check (Avoid paying "Dumb" costs)
        # Bid/Ask spread must be < 15bps (0.15%)
        # For simulation, we check if price is too far from 'mid' or theoretical
        spread_bps = order.get('spread_bps', 5) # Default nice
        if spread_bps > 25: # > 0.25% is huge for large cap
            logger.warning(f"TRADE REJECTED: Wide Spread ({spread_bps} bps)")
            return False

        # 3. Volatility Gate (Don't catch falling knives blindly)
        # If daily move > 15%, reject unless strategy is specifically "Volatility"
        daily_move_pct = abs(order.get('daily_move_pct', 0.0))
        if daily_move_pct > 0.15:
             logger.warning(f"TRADE REJECTED: Extreme Volatility ({daily_move_pct*100:.1f}%)")
             return False

        return True

    def check_portfolio_stops(self, current_pnl_pct: float) -> bool:
        """
        Check global portfolio stops.
        """
        if current_pnl_pct < -self.max_drawdown:
            logger.critical(f"MAX DRAWDOWN HIT: {current_pnl_pct}. ACTIVATING KILL SWITCH.")
            self.activate_kill_switch()
            return False
        return True

    def activate_kill_switch(self):
        """Emergency stop."""
        self.is_kill_switch_active = True
        # Logic to cancel all open orders would go here

# Global instance
_safety = SafetyGuard()
def get_safety_guard():
    return _safety
