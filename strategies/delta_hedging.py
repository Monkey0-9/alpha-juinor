"""
Delta Hedging Strategy
=======================

Dynamic delta-hedging and gamma scalping strategies for options portfolios.

Features:
- Continuous delta hedging
- Gamma scalping
- Vega hedging
- Position rebalancing optimization
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from derivatives.volatility_surface import BlackScholesModel

logger = logging.getLogger(__name__)


@dataclass
class OptionPosition:
    """An option position in the portfolio."""

    symbol: str
    strike: float
    expiry_days: int
    option_type: str  # 'call' or 'put'
    quantity: int  # Positive for long, negative for short
    entry_price: float
    current_price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


@dataclass
class HedgePosition:
    """A hedging position (usually in the underlying)."""

    symbol: str
    quantity: float  # Shares
    entry_price: float
    current_price: float = 0.0


@dataclass
class DeltaHedgingStrategy:
    """
    Dynamic delta-hedging strategy.

    Maintains delta-neutral portfolio by rebalancing the underlying position.
    """

    underlying_symbol: str
    options: List[OptionPosition] = field(default_factory=list)
    hedge_position: Optional[HedgePosition] = None

    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    rebalance_threshold: float = 0.1  # Rebalance when delta exceeds this
    transaction_cost_pct: float = 0.001  # 0.1% transaction cost

    def add_option(self, option: OptionPosition):
        """Add an option position to the portfolio."""
        self.options.append(option)
        logger.info(
            f"Added {option.quantity} x {option.option_type} "
            f"{option.strike} expiring in {option.expiry_days}d"
        )

    def calculate_portfolio_greeks(
        self, underlying_price: float, current_vol: float
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks.

        Args:
            underlying_price: Current underlying price
            current_vol: Current implied volatility

        Returns:
            Dictionary of portfolio Greeks
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        for option in self.options:
            T = option.expiry_days / 365.0

            # Calculate Greeks
            d1 = BlackScholesModel.d1(
                underlying_price,
                option.strike,
                T,
                self.risk_free_rate,
                self.dividend_yield,
                current_vol,
            )
            d2 = BlackScholesModel.d2(
                underlying_price,
                option.strike,
                T,
                self.risk_free_rate,
                self.dividend_yield,
                current_vol,
            )

            # Delta
            if option.option_type.lower() == "call":
                delta = np.exp(-self.dividend_yield * T) * norm.cdf(d1)
            else:
                delta = -np.exp(-self.dividend_yield * T) * norm.cdf(-d1)

            # Gamma (same for calls and puts)
            gamma = (
                np.exp(-self.dividend_yield * T)
                * norm.pdf(d1)
                / (underlying_price * current_vol * np.sqrt(T))
            )

            # Vega (same for calls and puts)
            vega = (
                underlying_price
                * np.exp(-self.dividend_yield * T)
                * norm.pdf(d1)
                * np.sqrt(T)
            ) / 100

            # Theta
            term1 = -(
                underlying_price
                * norm.pdf(d1)
                * current_vol
                * np.exp(-self.dividend_yield * T)
            ) / (2 * np.sqrt(T))
            term2 = self.dividend_yield * underlying_price * np.exp(
                -self.dividend_yield * T
            )
            term3 = self.risk_free_rate * option.strike * np.exp(
                -self.risk_free_rate * T
            )

            if option.option_type.lower() == "call":
                theta = (term1 + term2 * norm.cdf(d1) - term3 * norm.cdf(d2)) / 365
            else:
                theta = (term1 - term2 * norm.cdf(-d1) + term3 * norm.cdf(-d2)) / 365

            # Update option Greeks
            option.delta = delta
            option.gamma = gamma
            option.vega = vega
            option.theta = theta

            # Accumulate portfolio Greeks
            total_delta += delta * option.quantity
            total_gamma += gamma * option.quantity
            total_vega += vega * option.quantity
            total_theta += theta * option.quantity

        # Add hedge position delta
        if self.hedge_position:
            total_delta += self.hedge_position.quantity

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega": total_vega,
            "theta": total_theta,
        }

    def get_hedge_recommendation(
        self, underlying_price: float, current_vol: float
    ) -> Dict:
        """
        Get hedging recommendation to neutralize delta.

        Args:
            underlying_price: Current underlying price
            current_vol: Current implied volatility

        Returns:
            Dictionary with hedge recommendation
        """
        greeks = self.calculate_portfolio_greeks(underlying_price, current_vol)
        current_delta = greeks["delta"]

        # Calculate required hedge position
        target_hedge_quantity = -current_delta

        # Get current hedge quantity
        current_hedge_quantity = (
            self.hedge_position.quantity if self.hedge_position else 0.0
        )

        # Calculate adjustment needed
        adjustment = target_hedge_quantity - current_hedge_quantity

        # Check if rebalancing is needed
        should_rebalance = abs(current_delta) > self.rebalance_threshold

        # Calculate transaction cost
        transaction_cost = (
            abs(adjustment) * underlying_price * self.transaction_cost_pct
        )

        return {
            "current_delta": current_delta,
            "current_gamma": greeks["gamma"],
            "current_vega": greeks["vega"],
            "current_theta": greeks["theta"],
            "current_hedge_quantity": current_hedge_quantity,
            "target_hedge_quantity": target_hedge_quantity,
            "adjustment_needed": adjustment,
            "should_rebalance": should_rebalance,
            "transaction_cost": transaction_cost,
            "recommendation": (
                f"{'BUY' if adjustment > 0 else 'SELL'} "
                f"{abs(adjustment):.2f} shares at ${underlying_price:.2f}"
                if should_rebalance
                else "No rebalancing needed"
            ),
        }

    def execute_hedge(self, underlying_price: float, adjustment: float):
        """
        Execute hedge trade.

        Args:
            underlying_price: Current underlying price
            adjustment: Shares to buy (positive) or sell (negative)
        """
        if self.hedge_position is None:
            self.hedge_position = HedgePosition(
                symbol=self.underlying_symbol,
                quantity=adjustment,
                entry_price=underlying_price,
                current_price=underlying_price,
            )
        else:
            # Update position
            old_quantity = self.hedge_position.quantity
            new_quantity = old_quantity + adjustment

            # Update weighted average entry price
            if new_quantity != 0:
                total_cost = (
                    old_quantity * self.hedge_position.entry_price
                    + adjustment * underlying_price
                )
                self.hedge_position.entry_price = total_cost / new_quantity

            self.hedge_position.quantity = new_quantity
            self.hedge_position.current_price = underlying_price

        logger.info(
            f"Executed hedge: {adjustment:+.2f} shares at ${underlying_price:.2f}"
        )

    def get_pnl(self, underlying_price: float, current_vols: Dict[str, float]) -> Dict:
        """
        Calculate P&L breakdown.

        Args:
            underlying_price: Current underlying price
            current_vols: Current implied volatilities for each option

        Returns:
            Dictionary with P&L components
        """
        options_pnl = 0.0
        hedge_pnl = 0.0

        # Options P&L
        for option in self.options:
            T = option.expiry_days / 365.0
            vol = current_vols.get(
                f"{option.strike}_{option.expiry_days}", 0.3
            )  # Default vol

            current_price = BlackScholesModel.price(
                underlying_price,
                option.strike,
                T,
                self.risk_free_rate,
                self.dividend_yield,
                vol,
                option.option_type,
            )

            option.current_price = current_price
            pnl = (current_price - option.entry_price) * option.quantity
            options_pnl += pnl

        # Hedge P&L
        if self.hedge_position:
            self.hedge_position.current_price = underlying_price
            hedge_pnl = (
                underlying_price - self.hedge_position.entry_price
            ) * self.hedge_position.quantity

        total_pnl = options_pnl + hedge_pnl

        return {
            "options_pnl": options_pnl,
            "hedge_pnl": hedge_pnl,
            "total_pnl": total_pnl,
            "pnl_pct": (
                total_pnl
                / (sum(abs(o.entry_price * o.quantity) for o in self.options) + 1e-6)
                * 100
            ),
        }


from scipy.stats import norm


class GammaScalpingStrategy(DeltaHedgingStrategy):
    """
    Gamma scalping strategy.

    Profits from realized volatility being higher than implied volatility
    through continuous delta hedging.
    """

    def __init__(self, *args, gamma_threshold: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_threshold = gamma_threshold
        self.scalping_history: List[Dict] = []

    def evaluate_scalping_opportunity(
        self, underlying_price: float, current_vol: float, realized_vol: float
    ) -> Dict:
        """
        Evaluate gamma scalping opportunity.

        Args:
            underlying_price: Current underlying price
            current_vol: Implied volatility
            realized_vol: Realized historical volatility

        Returns:
            Dictionary with scalping analysis
        """
        greeks = self.calculate_portfolio_greeks(underlying_price, current_vol)
        gamma = greeks["gamma"]

        # Gamma scalping profit potential
        vol_diff = realized_vol - current_vol
        expected_scalping_pnl = 0.5 * gamma * (underlying_price**2) * (
            realized_vol**2 - current_vol**2
        )

        opportunity = {
            "gamma": gamma,
            "implied_vol": current_vol,
            "realized_vol": realized_vol,
            "vol_edge": vol_diff,
            "expected_pnl": expected_scalping_pnl,
            "is_profitable": vol_diff > 0 and abs(gamma) > self.gamma_threshold,
        }

        self.scalping_history.append(
            {"timestamp": datetime.now(), **opportunity}
        )

        return opportunity
