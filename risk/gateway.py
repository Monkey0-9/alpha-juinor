"""
ELITE RISK GATEWAY
==================

Pre-Trade Risk Checks.
Must pass BEFORE order is routed to SOR.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger("RISK_GATEWAY")

class RiskGateway:
    def __init__(self):
        self.max_order_value = 1_000_000 # $1M Limit
        self.restricted_list = ["GME", "AMC"] # No Meme Stocks

    def check_order(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Validate order against strict risk limits.
        """
        # 1. Restricted Check
        if symbol in self.restricted_list:
            logger.warning(f"RISK REJECT: {symbol} is on Restricted List.")
            return False

        # 2. Notional Check
        notional = abs(quantity * price)
        if notional > self.max_order_value:
            logger.warning(f"RISK REJECT: Notional ${notional:,.0f} exceeds limit ${self.max_order_value:,.0f}.")
            return False

        # 3. Fat Finger Check (Sanity)
        if quantity == 0:
            return False

        return True
