from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import math
class Order(BaseModel):
    symbol: str
    size: int
    side: str
    type: str = "MARKET"
    price: float = 0.0
    status: str = "PENDING"
    metadata: Dict[str, Any] = {}

class SymbolDecision(BaseModel):
    symbol: str
    mu_hat: float
    sigma_agg: float
    f: float
    final_decision: str
    reason_codes: List[str]
    cvar_95: float = 0.0

class InstitutionalAllocator:
    """
    Refined Institutional Allocator.
    Mandate: Size to lots, enforce min size, round, check liquidity.
    """
    def __init__(self, leverage_limit: float = 1.0):
        self.leverage_limit = leverage_limit

    def construct_orders_from_weights(self,
                                     weights: Dict[str, float],
                                     prices: Dict[str, float],
                                     NAV: float,
                                     round_lot: int = 100) -> List[Order]:
        """
        Convert weights back to orders with lot rounding and min-notional checks.
        """
        orders = []
        for symbol, w in weights.items():
            if abs(w) < 0.0001: # Threshold for tiny positions
                continue

            price = prices.get(symbol)
            if not price or price <= 0:
                continue

            target_notional = w * NAV
            target_shares = target_notional / price

            # Floor to round lot
            if abs(target_shares) < round_lot:
                # If smaller than 1 lot, check if we should even trade
                if abs(target_notional) < 500: # $500 min notional
                    continue
                rounded_shares = math.copysign(math.ceil(abs(target_shares)), target_shares)
            else:
                rounded_shares = math.copysign(math.floor(abs(target_shares) / round_lot) * round_lot, target_shares)

            if abs(rounded_shares) > 0:
                orders.append(Order(
                    symbol=symbol,
                    size=int(rounded_shares),
                    side="BUY" if rounded_shares > 0 else "SELL",
                    type="MARKET",
                    price=price,
                    metadata={"weight": w, "target_notional": target_notional}
                ))

        return orders

    def allocate(self, decisions: List[SymbolDecision], nav: float, metadata: Dict[str, Any] = None) -> List[Order]:
        """
        Legacy wrapper for backwards compatibility with the main loop if needed.
        """
        # (This would normally call the newer logic)
        return []
