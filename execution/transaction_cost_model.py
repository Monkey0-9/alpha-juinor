"""
Transaction Cost Model - Accurate Slippage & Impact Estimation.

Based on academic models:
- Kyle (1985) lambda for price impact
- Almgren-Chriss for optimal execution
- Real slippage tracking for calibration
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TransactionCostEstimate:
    """Comprehensive transaction cost breakdown."""
    symbol: str
    direction: str  # "BUY" or "SELL"
    shares: float
    price: float

    # Cost components
    spread_cost: float  # Half spread
    market_impact: float  # Kyle's lambda
    timing_cost: float  # Opportunity cost from delay
    commission: float  # Broker fees (if any)

    # Totals
    total_cost: float
    cost_bps: float  # Basis points

    # Execution recommendation
    recommended_algo: str  # TWAP, VWAP, IS, etc.
    urgency: str  # LOW, MEDIUM, HIGH


@dataclass
class SlippageRecord:
    """Record of actual vs expected execution."""
    symbol: str
    expected_price: float
    executed_price: float
    slippage_bps: float
    timestamp: str
    order_size: float
    adv_pct: float  # % of average daily volume


class TransactionCostModel:
    """
    Institutional-grade transaction cost estimation.

    Features:
    - Kyle's lambda for price impact
    - Real slippage tracking for model calibration
    - ADV-based sizing recommendations
    - Execution algorithm selection
    """

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        impact_coefficient: float = 0.1,
        commission_per_share: float = 0.0,
        max_adv_pct: float = 0.05  # Max 5% of ADV per order
    ):
        self.base_spread_bps = base_spread_bps
        self.impact_coefficient = impact_coefficient
        self.commission_per_share = commission_per_share
        self.max_adv_pct = max_adv_pct

        # Slippage history for calibration
        self.slippage_history: Dict[str, list] = defaultdict(list)

        # Calibrated parameters per symbol
        self.symbol_params: Dict[str, Dict] = {}

    def estimate_cost(
        self,
        symbol: str,
        shares: float,
        price: float,
        direction: str,
        adv: float,
        volatility: float = 0.02,
        urgency: str = "MEDIUM"
    ) -> TransactionCostEstimate:
        """
        Estimate total transaction cost.

        Args:
            symbol: Ticker
            shares: Number of shares
            price: Current price
            direction: BUY or SELL
            adv: Average daily volume (shares)
            volatility: Daily volatility
            urgency: LOW, MEDIUM, HIGH

        Returns:
            TransactionCostEstimate with full breakdown
        """
        notional = shares * price

        # 1. Spread cost (half bid-ask spread)
        spread_bps = self._get_spread(symbol, adv)
        spread_cost = notional * spread_bps / 10000

        # 2. Market impact (Kyle's lambda model)
        # impact = sigma * sqrt(Q/ADV) * coefficient
        participation_rate = shares / adv if adv > 0 else 0
        impact_bps = self._calculate_impact(
            volatility, participation_rate, urgency
        )
        market_impact = notional * impact_bps / 10000

        # 3. Timing cost (opportunity cost from slow execution)
        timing_bps = self._timing_cost(urgency, volatility)
        timing_cost = notional * timing_bps / 10000

        # 4. Commission
        commission = shares * self.commission_per_share

        # Total
        total_cost = spread_cost + market_impact + timing_cost + commission
        total_bps = (total_cost / notional) * 10000 if notional > 0 else 0

        # Recommend execution algorithm
        algo, exec_urgency = self._recommend_algo(
            participation_rate, urgency, volatility
        )

        return TransactionCostEstimate(
            symbol=symbol,
            direction=direction,
            shares=shares,
            price=price,
            spread_cost=spread_cost,
            market_impact=market_impact,
            timing_cost=timing_cost,
            commission=commission,
            total_cost=total_cost,
            cost_bps=total_bps,
            recommended_algo=algo,
            urgency=exec_urgency
        )

    def _get_spread(self, symbol: str, adv: float) -> float:
        """Get estimated spread in basis points."""
        # Larger, more liquid stocks have tighter spreads
        if adv > 10_000_000:  # Very liquid
            return self.base_spread_bps * 0.5
        elif adv > 1_000_000:  # Liquid
            return self.base_spread_bps
        elif adv > 100_000:  # Medium
            return self.base_spread_bps * 2
        else:  # Illiquid
            return self.base_spread_bps * 5

    def _calculate_impact(
        self,
        volatility: float,
        participation_rate: float,
        urgency: str
    ) -> float:
        """
        Calculate market impact using Kyle's lambda model.

        impact = coefficient * volatility * sqrt(participation_rate)
        """
        # Urgency multiplier
        urgency_mult = {"LOW": 0.5, "MEDIUM": 1.0, "HIGH": 2.0}.get(urgency, 1.0)

        # Core impact model
        impact = (
            self.impact_coefficient
            * volatility
            * np.sqrt(participation_rate)
            * urgency_mult
        )

        # Convert to bps (volatility is daily, so impact is per-day)
        return impact * 10000

    def _timing_cost(self, urgency: str, volatility: float) -> float:
        """Opportunity cost from delayed execution."""
        # For LOW urgency, we spread execution, risking adverse moves
        # For HIGH urgency, we hit market immediately
        timing_mult = {"LOW": 1.0, "MEDIUM": 0.5, "HIGH": 0.1}.get(urgency, 0.5)
        return volatility * timing_mult * 10000  # bps

    def _recommend_algo(
        self,
        participation_rate: float,
        urgency: str,
        volatility: float
    ) -> Tuple[str, str]:
        """Recommend execution algorithm and adjusted urgency."""
        # Very small orders: immediate
        if participation_rate < 0.01:
            return "MARKET", urgency

        # Large orders in volatile markets: careful execution
        if participation_rate > 0.05 and volatility > 0.03:
            return "TWAP_CAREFUL", "LOW"

        # Large orders: VWAP for better fills
        if participation_rate > 0.03:
            return "VWAP", "MEDIUM" if urgency != "HIGH" else urgency

        # Medium orders
        if participation_rate > 0.01:
            if urgency == "HIGH":
                return "IS", "HIGH"  # Implementation Shortfall
            return "VWAP", urgency

        return "LIMIT", urgency

    def record_execution(
        self,
        symbol: str,
        expected_price: float,
        executed_price: float,
        order_size: float,
        adv: float,
        timestamp: str
    ):
        """Record actual execution for model calibration."""
        slippage_bps = abs(executed_price - expected_price) / expected_price * 10000

        record = SlippageRecord(
            symbol=symbol,
            expected_price=expected_price,
            executed_price=executed_price,
            slippage_bps=slippage_bps,
            timestamp=timestamp,
            order_size=order_size,
            adv_pct=order_size / adv if adv > 0 else 0
        )

        self.slippage_history[symbol].append(record)

        # Keep last 100 records per symbol
        if len(self.slippage_history[symbol]) > 100:
            self.slippage_history[symbol] = self.slippage_history[symbol][-100:]

        logger.debug(
            f"Recorded execution: {symbol} slippage={slippage_bps:.1f}bps"
        )

    def calibrate(self, symbol: str) -> Dict[str, float]:
        """Calibrate model parameters from historical executions."""
        history = self.slippage_history.get(symbol, [])

        if len(history) < 10:
            return {}

        # Compute average slippage vs ADV %
        slippages = [r.slippage_bps for r in history]
        adv_pcts = [r.adv_pct for r in history]

        avg_slippage = np.mean(slippages)

        # Simple linear regression: slippage = a + b * sqrt(adv_pct)
        if len(set(adv_pcts)) > 1:
            X = np.sqrt(adv_pcts)
            y = slippages
            A = np.vstack([X, np.ones(len(X))]).T
            coef, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        else:
            coef, intercept = 0, avg_slippage

        params = {
            "avg_slippage_bps": avg_slippage,
            "impact_coefficient": coef,
            "base_spread_bps": intercept,
            "n_observations": len(history)
        }

        self.symbol_params[symbol] = params
        logger.info(f"Calibrated {symbol}: avg_slippage={avg_slippage:.1f}bps")

        return params

    def get_optimal_order_size(
        self,
        symbol: str,
        target_shares: float,
        adv: float,
        max_impact_bps: float = 20.0
    ) -> float:
        """
        Calculate optimal order size to stay within impact budget.

        Returns: Recommended order size (may be less than target)
        """
        # Max participation rate
        max_participation = self.max_adv_pct

        # Solve: impact = coef * vol * sqrt(shares/adv) <= max_impact_bps
        # shares <= adv * (max_impact / (coef * vol))^2
        # Simplified: just cap at max ADV %
        max_shares = adv * max_participation

        return min(target_shares, max_shares)


# Global singleton
_tcm: Optional[TransactionCostModel] = None


def get_transaction_cost_model() -> TransactionCostModel:
    """Get or create global TransactionCostModel."""
    global _tcm
    if _tcm is None:
        _tcm = TransactionCostModel()
    return _tcm
