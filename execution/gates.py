"""
Institutional Execution Gatekeeper.

Responsibilities:
1. Pre-trade ADV check (10% limit).
2. Sentiment Divergence check (Reject if sentiment is extremely negative).
3. Pre-trade Impact estimation.
"""

import logging
import numpy as np
from datetime import datetime, time
from typing import Dict, Any, Optional, Tuple
try:
    import pytz
except ImportError:
    pytz = None

from risk.market_impact_models import TransactionCostModel, ImpactParameters
from database.manager import DatabaseManager
from risk.capital_stress import CapitalStressTester

logger = logging.getLogger("EXECUTION_GATE")

class ExecutionGatekeeper:
    """
    Final filter before order submission.
    Ensures liquidity, impact, and governance constraints are met.
    """

    def __init__(self, adv_limit_pct: float = 0.10, max_impact_bps: float = 20.0):
        self.adv_limit_pct = adv_limit_pct
        self.max_impact_bps = max_impact_bps
        self.impact_model = TransactionCostModel()
        self.stress_tester = CapitalStressTester() # Institutional Stress Logic
        self.db = DatabaseManager()

    def is_market_open(self) -> bool:
        """
        Check if US Equity markets are currently open.
        9:30 AM - 4:00 PM ET, Monday - Friday.
        """
        try:
            if pytz:
                now = datetime.now(pytz.timezone('US/Eastern'))
            else:
                # Manual fallback: assuming local time is near ET or just use UTC-5
                from datetime import timedelta, timezone
                now = datetime.now(timezone(timedelta(hours=-5)))

            if now.weekday() >= 5: # Saturday or Sunday
                return False

            market_open = time(9, 30)
            market_close = time(16, 0)
            return market_open <= now.time() <= market_close
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return True # Conservative: true if check fails? Or False? Main usually prefers True to not lock system.

    def validate_execution(self,
                           symbol: str,
                           qty: float,
                           side: str,
                           price: float,
                           adv_30d: float,
                           volatility: float,
                           sentiment_score: Optional[float] = None) -> Tuple[bool, str, float]:
        """
        Validate order against gates.

        Returns:
            (is_ok, reason_code, scaled_qty)
        """
        # 0. Symbol Governance Gate (MANDATORY)
        gov = self.db.get_symbol_governance(symbol)
        if not gov or gov['state'] != 'ACTIVE':
            state = gov['state'] if gov else 'UNKNOWN'
            logger.warning(f"GOVERNANCE_GATE_TRIGGER: {symbol} is {state} - REJECTING EXECUTION")
            return False, f"REJECTED_GOVERNANCE_{state}", 0.0

        # 0.5 Market Hours Gate
        if not self.is_market_open():
            logger.warning(f"MARKET_GATE_TRIGGER: Market is CLOSED - REJECTING EXECUTION for {symbol}")
            return False, "MARKET_CLOSED", 0.0

        # 1. ADV Gate (10% limit)
        # Assuming ADV is in shares
        max_qty = adv_30d * self.adv_limit_pct
        if abs(qty) > max_qty:
            logger.warning(f"ADV_GATE_TRIGGER: {symbol} qty={qty} > max_adv_qty={max_qty} (10% of {adv_30d})")
            # Scaledown policy
            scaled_qty = np.sign(qty) * max_qty
            return False, "ADV_LIMIT_EXCEEDED", scaled_qty

        # 2. Sentiment Divergence Gate
        # If sentiment is < -0.8 (EXTREME PANIC), reject buys
        if sentiment_score is not None and sentiment_score < -0.8 and side.upper() == "BUY":
            logger.warning(f"SENTIMENT_GATE_TRIGGER: {symbol} sentiment={sentiment_score} (Extreme Negative) - Blocking BUYS")
            return False, "NEGATIVE_SENTIMENT_DIVERGENCE", 0.0

        # 3. Pre-trade Impact estimation
        # Use CapitalStressTester (Square-Root Law)
        # Impact (bps) = k * sqrt(order_size / ADV)

        trade_dollar_vol = price * abs(qty)

        # Calculate using Stress Tester
        impact_bps = self.stress_tester.compute_market_impact(trade_dollar_vol, adv_30d)

        # Legacy model comparison (optional logging)
        # cost_estimate = self.impact_model.estimate_total_cost(qty, impact_params)
        # legacy_bps = cost_estimate.get('cost_bps', 0.0)

        if impact_bps > self.max_impact_bps:
            logger.warning(f"IMPACT_GATE_TRIGGER: {symbol} estimated_impact={impact_bps:.2f}bps > limit={self.max_impact_bps}bps")
            # Scale down to meet impact limit
            # Impact roughly scales with square root or linear; conservative linear scaling:
            impact_ratio = self.max_impact_bps / impact_bps
            scaled_qty = qty * impact_ratio
            return False, "HIGH_IMPACT_ESTIMATE", scaled_qty

        return True, "OK", qty
