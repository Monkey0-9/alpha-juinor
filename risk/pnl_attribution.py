"""
risk/pnl_attribution.py

Section D: PnL Attribution.
Decomposes realized PnL into components.
"""

from dataclasses import dataclass
from typing import List, Any
# from execution.fill_simulator import Fill -> Forbidden
# We assume fills are passed as objects with .qty, .price, .commission, .symbol
# Or we fix the architecture correctly by moving Fill to contracts.
# Quick fix: type hint as Any or Protocol.
from typing import Protocol

class FillProtocol(Protocol):
    qty: float
    price: float
    commission: float
    symbol: str

@dataclass
class AttributionRecord:
    symbol: str
    total_pnl: float
    market_pnl: float
    alpha_pnl: float
    execution_loss: float
    fees: float

class PnLAttribution:
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def attribute(self,
                  fills: List[FillProtocol],
                  arrival_price: float,

                  exit_price: float,
                  benchmark_return: float,
                  position_side: str) -> AttributionRecord:

        # Aggregate fills
        total_qty = sum(f.qty for f in fills)
        total_cost = sum(f.price * f.qty for f in fills)
        total_fees = sum(f.commission for f in fills)

        avg_price = total_cost / total_qty if total_qty > 0 else 0

        # Total PnL (Realized)
        direction = 1 if position_side == "BUY" else -1
        gross_pnl = (exit_price - avg_price) * total_qty * direction
        net_pnl = gross_pnl - total_fees

        # Execution Loss (Implementation Shortfall)
        # Difference between Arrival Price and Avg Fill Price
        # Slippage cost
        exec_diff = (avg_price - arrival_price) * direction
        # Positive diff for BUY means we paid more -> Loss
        # Positive diff for SELL means we sold for more -> Gain (Negative Loss)
        execution_loss = exec_diff * total_qty
        # Note: If we paid more than arrival, execution_loss is positive cost.

        # Theoretical PnL (If filled at Arrival)
        theoretical_pnl = (exit_price - arrival_price) * total_qty * direction

        # Market PnL (Beta * Bench)
        # Portion of theoretical PnL explained by market
        market_pnl_pct = self.beta * benchmark_return
        market_pnl_dollar = arrival_price * total_qty * market_pnl_pct * direction

        # Alpha PnL (Residual)
        alpha_pnl = theoretical_pnl - market_pnl_dollar

        return AttributionRecord(
            symbol=fills[0].symbol if fills else "UNKNOWN",
            total_pnl=net_pnl,
            market_pnl=market_pnl_dollar,
            alpha_pnl=alpha_pnl,
            execution_loss=execution_loss,
            fees=total_fees
        )
