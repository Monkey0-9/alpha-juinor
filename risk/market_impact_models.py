from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ImpactParameters:
    """Parameters for market impact estimation."""

    volatility: float
    adv: float
    market_cap: float
    liquidity_score: float
    order_size_pct: float
    time_horizon: float
    risk_aversion: float


class TransactionCostModel:
    """
    Models for estimating market impact and slippage.
    """

    @staticmethod
    def almgren_chriss_slippage(
        qty: float, adv: float, volatility: float, price: float, eta: float = 0.1
    ) -> float:
        """
        Estimate slippage using simplified Almgren-Chriss square root law.

        Slippage (bps) ~ eta * volatility * sqrt(qty / adv)

        Args:
            qty: Order quantity
            adv: Average Daily Volume
            volatility: Daily volatility (decimal)
            price: Current price
            eta: Impact coefficient (tuned parameter)

        Returns:
            Estimated slippage cost in Dollars per share
        """
        if adv <= 0 or qty <= 0:
            return 0.0

        # Participation rate
        participation = qty / adv

        # Impact cost (percentage of price)
        # Linear/Sqrt hybrid often used. Here we use Sqrt law for general liquidity.
        impact_pct = eta * volatility * np.sqrt(participation)

        return price * impact_pct

    @staticmethod
    def estimate_cost(
        qty: float,
        price: float,
        slippage_per_share: float,
        commission_per_share: float = 0.005,
    ) -> float:
        """Total transaction cost estimate."""
        return (qty * slippage_per_share) + (qty * commission_per_share)

    def estimate_total_cost(
        self, qty: float, params: ImpactParameters
    ) -> Dict[str, float]:
        """
        Estimate total cost in bps for a given quantity and parameters.
        Matches the interface expected by ExecutionGatekeeper.
        """
        # Simple square root law approximation: cost_bps = volatility * sqrt(qty / adv) * 10000
        if params.adv <= 0:
            return {"cost_bps": 0.0}

        participation = abs(qty) / params.adv
        impact_pct = 0.1 * params.volatility * np.sqrt(participation)
        cost_bps = impact_pct * 10000

        return {"cost_bps": cost_bps}


# Alias for compatibility with new execution engine
MarketImpactModel = TransactionCostModel
