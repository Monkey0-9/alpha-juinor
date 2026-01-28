"""
Market Impact Models.

This module provides market impact and slippage models including:
- Almgren-Chriss optimal execution model
- Temporary vs permanent impact decomposition
- Volatility-adjusted sizing
- Liquidity estimation
"""

import os
import sys
import math
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ImpactEstimate:
    """Market impact estimation result."""
    impact_bps: float
    market_impact: float
    temporary_impact: float
    permanent_impact: float
    slippage_estimate: float
    execution_time_minutes: float
    participation_rate: float


class MarketImpactModel:
    """
    Market impact model based on Almgren-Chriss framework.

    Total Impact = sigma * sqrt(Q/V) * (epsilon + eta * Q/V)
    """

    def __init__(self,
                 temporary_coefficient: float = 0.1,
                 permanent_coefficient: float = 0.05,
                 volatility_scaling: bool = True,
                 liquidity_adjusted: bool = True):
        self.temporary_coefficient = temporary_coefficient
        self.permanent_coefficient = permanent_coefficient
        self.volatility_scaling = volatility_scaling
        self.liquidity_adjusted = liquidity_adjusted

        self.default_daily_volatility = 0.02
        self.default_adv_pct = 0.001

        logger.info(f"MarketImpactModel initialized - temp_coef={temporary_coefficient}, perm_coef={permanent_coefficient}")

    def estimate_impact(self,
                       symbol: str,
                       side: str,
                       quantity: float,
                       order_type: str = "MARKET",
                       price: float = 100.0,
                       volatility: Optional[float] = None,
                       adv: Optional[float] = None,
                       participation_rate: float = 0.1,
                       execution_time_hours: float = 1.0) -> ImpactEstimate:
        """Estimate market impact for an order."""
        vol = volatility or self.default_daily_volatility

        if adv is None:
            adv = quantity / participation_rate if participation_rate > 0 else quantity * 10

        participation = quantity / adv if adv > 0 else self.default_adv_pct

        optimal_eta = self.permanent_coefficient * math.sqrt(participation)
        optimal_epsilon = self.temporary_coefficient * math.sqrt(participation)

        vol_scaled = vol * math.sqrt(252) if self.volatility_scaling else vol
        participation_scaled = math.sqrt(participation)

        temporary = vol_scaled * participation_scaled * optimal_epsilon * price
        permanent = vol_scaled * participation * optimal_eta * price

        total_impact = temporary + permanent

        impact_bps = (total_impact / price) * 10000 if price > 0 else 0

        if order_type == "MARKET":
            slippage_bps = impact_bps * 1.2
        elif order_type == "LIMIT":
            slippage_bps = impact_bps * 0.3
        else:
            slippage_bps = impact_bps * 0.8

        slippage_estimate = (slippage_bps / 10000) * price * quantity

        optimal_time = min(6.0, max(0.25, math.log(participation + 1) * execution_time_hours))

        optimal_participation = min(0.25, math.sqrt(1e-5) / (vol * math.sqrt(adv)))

        return ImpactEstimate(
            impact_bps=impact_bps,
            market_impact=total_impact,
            temporary_impact=temporary,
            permanent_impact=permanent,
            slippage_estimate=slippage_estimate,
            execution_time_minutes=optimal_time * 60,
            participation_rate=optimal_participation,
        )

    def calculate_optimal_execution(self,
                                    symbol: str,
                                    side: str,
                                    quantity: float,
                                    price: float,
                                    volatility: float = 0.02,
                                    adv: float = 1000000,
                                    risk_aversion: float = 1e-5) -> Dict[str, Any]:
        """Calculate optimal execution trajectory using Almgren-Chriss."""
        vol_daily = volatility * math.sqrt(1/252)

        eta = self.permanent_coefficient * math.sqrt(quantity / adv)
        epsilon = self.temporary_coefficient * math.sqrt(quantity / adv)

        lambda_r = risk_aversion

        numerator = lambda_r * vol_daily**2 * (1/252)
        denominator = epsilon**2 + 4 * lambda_r * vol_daily**2 * eta * (1/252)

        optimal_lambda = math.sqrt(numerator / denominator) if denominator > 0 else 0.1

        trajectory = []
        remaining = quantity
        time_step = 0

        while remaining > 0 and time_step < 100:
            trade = min(remaining, optimal_lambda * quantity)

            trajectory.append({
                "time_step": time_step,
                "trade_quantity": trade,
                "remaining_quantity": remaining - trade,
                "execution_time_hours": time_step * 24 / 252,
            })

            remaining -= trade
            time_step += 1

        total_cost = self._calculate_total_cost(
            quantity, price, vol_daily, eta, epsilon, trajectory
        )

        var_95 = 1.65 * vol_daily * price * math.sqrt(sum(t["trade_quantity"]**2 for t in trajectory))

        return {
            "symbol": symbol,
            "side": side,
            "total_quantity": quantity,
            "optimal_trajectory": trajectory,
            "expected_cost": total_cost,
            "cost_bps": (total_cost / (price * quantity)) * 10000 if quantity > 0 else 0,
            "var_95": var_95,
            "optimal_duration_days": len(trajectory),
            "parameters": {
                "volatility": volatility,
                "adv": adv,
                "risk_aversion": risk_aversion,
                "eta": eta,
                "epsilon": epsilon,
            }
        }

    def _calculate_total_cost(self,
                              quantity: float,
                              price: float,
                              vol_daily: float,
                              eta: float,
                              epsilon: float,
                              trajectory: List[Dict]) -> float:
        """Calculate total expected execution cost."""
        total_cost = 0.0
        cumulative = 0.0

        for step in trajectory:
            trade_qty = step["trade_quantity"]
            temp_impact = epsilon * vol_daily * math.sqrt(trade_qty) * price
            perm_impact = eta * cumulative * vol_daily * price

            step_cost = temp_impact + perm_impact
            total_cost += step_cost
            cumulative += trade_qty

        return total_cost

    def calibrate_from_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calibrate model parameters from historical trade data."""
        if len(trades) < 10:
            logger.warning("Not enough trades for calibration")
            return {
                "temporary_coefficient": self.temporary_coefficient,
                "permanent_coefficient": self.permanent_coefficient,
            }

        x_values = []
        y_values = []

        for trade in trades:
            try:
                participation = trade["quantity"] / trade.get("adv", 1000000)
                vol = trade.get("volatility", self.default_daily_volatility)

                if participation > 0 and participation < 1:
                    impact = trade.get("actual_slippage", 0) / trade.get("price", 100)
                    x = math.sqrt(participation) * vol
                    y = impact / vol if vol > 0 else 0

                    if x > 0:
                        x_values.append(x)
                        y_values.append(y)
            except (KeyError, ZeroDivisionError):
                continue

        if len(x_values) < 5:
            return {
                "temporary_coefficient": self.temporary_coefficient,
                "permanent_coefficient": self.permanent_coefficient,
            }

        avg_ratio = sum(y/x for x, y in zip(x_values, y_values) if x > 0) / len(x_values)

        calibrated_temp = avg_ratio * 0.5
        calibrated_perm = avg_ratio * 0.5

        self.temporary_coefficient = max(0.01, min(0.5, calibrated_temp))
        self.permanent_coefficient = max(0.01, min(0.3, calibrated_perm))

        logger.info(f"Calibrated: temp_coef={self.temporary_coefficient:.4f}, perm_coef={self.permanent_coefficient:.4f}")

        return {
            "temporary_coefficient": self.temporary_coefficient,
            "permanent_coefficient": self.permanent_coefficient,
        }


class LiquidityEstimator:
    """Estimate market liquidity for sizing decisions."""

    def __init__(self):
        self._volume_cache: Dict[str, float] = {}
        self._spread_cache: Dict[str, float] = {}

    def estimate_effective_spread(self, symbol: str, price: float) -> float:
        """Estimate effective spread as percentage."""
        vol = self._volume_cache.get(symbol, 1000000)

        if vol > 0:
            spread_pct = 0.0005 + 0.01 / math.sqrt(vol / 1e6)
        else:
            spread_pct = 0.001

        return min(spread_pct, 0.01)

    def estimate_market_depth(self, symbol: str, price: float) -> Dict[str, float]:
        """Estimate market depth at various levels."""
        base_volume = self._volume_cache.get(symbol, 1000000)

        return {
            "top_of_book": base_volume * 0.1,
            "depth_10bps": base_volume * 0.2,
            "depth_25bps": base_volume * 0.5,
            "depth_50bps": base_volume * 0.8,
        }

    def get_max_order_size(self,
                           symbol: str,
                           price: float,
                           max_impact_bps: float = 10,
                           max_participation: float = 0.25) -> float:
        """Calculate maximum safe order size."""
        adv = self._volume_cache.get(symbol, 1000000) * price

        temp_coef = 0.1

        if temp_coef > 0:
            impact_factor = max_impact_bps / temp_coef / 10000
            impact_shares = adv * impact_factor**2
        else:
            impact_shares = adv * max_participation

        return min(adv * max_participation, impact_shares)


def get_impact_model() -> MarketImpactModel:
    """Get market impact model singleton."""
    return MarketImpactModel()


def get_liquidity_estimator() -> LiquidityEstimator:
    """Get liquidity estimator singleton."""
    return LiquidityEstimator()

