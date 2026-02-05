"""
Smart Order Router (SOR) - Execution as Alpha
=================================================

Elite-tier execution: turn order routing into alpha.

Features:
1. Dark pool vs. lit routing decisions
2. Exchange rebate optimization
3. Liquidity aggregation across venues
4. Execution quality analysis
5. Maker/taker fee optimization

Execution is not just cost. It's alpha.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import threading

logger = logging.getLogger(__name__)


class Venue(Enum):
    """Trading venues."""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    ARCA = "ARCA"
    BATS = "BATS"
    IEX = "IEX"
    EDGX = "EDGX"

    # Dark pools
    CROSSFINDER = "CROSSFINDER"  # Credit Suisse
    SIGMA_X = "SIGMA_X"  # Goldman
    LIQUIDNET = "LIQUIDNET"
    POSIT = "POSIT"  # ITG
    LEVEL_ATS = "LEVEL_ATS"  # Citadel

    # Internalization
    INTERNAL = "INTERNAL"


class OrderUrgency(Enum):
    """Order urgency level."""
    PASSIVE = "PASSIVE"  # Take time, minimize impact
    NEUTRAL = "NEUTRAL"  # Balance
    AGGRESSIVE = "AGGRESSIVE"  # Fast execution priority


@dataclass
class VenueStats:
    """Statistics for a venue."""
    venue: Venue

    # Fill rates
    fill_rate: float  # % of orders filled
    avg_fill_time_ms: float

    # Pricing
    taker_fee: float  # Fee for taking liquidity
    maker_rebate: float  # Rebate for providing liquidity

    # Quality
    price_improvement_bps: float  # Basis points of improvement
    information_leakage: float  # 0-1, lower is better

    # Liquidity
    avg_displayed_size: int
    hidden_liquidity_pct: float

    # For dark pools
    is_dark: bool = False
    min_order_size: int = 0


# Venue characteristics (simplified)
VENUE_STATS = {
    Venue.NYSE: VenueStats(
        venue=Venue.NYSE,
        fill_rate=0.85,
        avg_fill_time_ms=50,
        taker_fee=0.0030,  # 30 bps
        maker_rebate=0.0020,  # 20 bps rebate
        price_improvement_bps=0.5,
        information_leakage=0.6,
        avg_displayed_size=500,
        hidden_liquidity_pct=0.1,
        is_dark=False
    ),
    Venue.NASDAQ: VenueStats(
        venue=Venue.NASDAQ,
        fill_rate=0.88,
        avg_fill_time_ms=30,
        taker_fee=0.0030,
        maker_rebate=0.0020,
        price_improvement_bps=0.3,
        information_leakage=0.55,
        avg_displayed_size=400,
        hidden_liquidity_pct=0.15,
        is_dark=False
    ),
    Venue.IEX: VenueStats(
        venue=Venue.IEX,
        fill_rate=0.70,
        avg_fill_time_ms=350,  # Speed bump!
        taker_fee=0.0009,  # Lower fee
        maker_rebate=0.0000,  # No maker rebate
        price_improvement_bps=2.0,  # Better improvement
        information_leakage=0.2,  # Less leakage
        avg_displayed_size=300,
        hidden_liquidity_pct=0.05,
        is_dark=False
    ),
    Venue.CROSSFINDER: VenueStats(
        venue=Venue.CROSSFINDER,
        fill_rate=0.35,  # Lower fill rate
        avg_fill_time_ms=100,
        taker_fee=0.0005,  # Very low
        maker_rebate=0.0000,
        price_improvement_bps=5.0,  # Best improvement
        information_leakage=0.1,  # Minimal leakage
        avg_displayed_size=0,
        hidden_liquidity_pct=1.0,
        is_dark=True,
        min_order_size=100
    ),
    Venue.SIGMA_X: VenueStats(
        venue=Venue.SIGMA_X,
        fill_rate=0.30,
        avg_fill_time_ms=80,
        taker_fee=0.0005,
        maker_rebate=0.0000,
        price_improvement_bps=4.5,
        information_leakage=0.15,
        avg_displayed_size=0,
        hidden_liquidity_pct=1.0,
        is_dark=True,
        min_order_size=100
    ),
    Venue.LIQUIDNET: VenueStats(
        venue=Venue.LIQUIDNET,
        fill_rate=0.20,  # Block-only, lower fill
        avg_fill_time_ms=500,
        taker_fee=0.0010,
        maker_rebate=0.0000,
        price_improvement_bps=8.0,  # Great for blocks
        information_leakage=0.05,  # Minimal
        avg_displayed_size=0,
        hidden_liquidity_pct=1.0,
        is_dark=True,
        min_order_size=5000  # Large blocks only
    )
}


@dataclass
class RoutingDecision:
    """Decision on how to route an order."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: int

    # Primary venue
    primary_venue: Venue
    primary_quantity: int

    # Secondary venues
    secondary_venues: List[Tuple[Venue, int]]

    # Expected metrics
    expected_fill_rate: float
    expected_cost_bps: float
    expected_fill_time_ms: float

    # Order type
    is_limit: bool
    limit_price: Optional[float]

    # Reason
    routing_reason: str


@dataclass
class ExecutionReport:
    """Report of executed order."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: int

    # Fills
    fills: List[Dict[str, Any]]
    total_filled: int
    avg_fill_price: float

    # Comparison
    arrival_price: float
    implementation_shortfall_bps: float

    # Costs
    total_fees: float
    total_rebates: float
    net_cost: float

    # Quality metrics
    price_improvement_bps: float
    fill_rate: float
    avg_fill_time_ms: float


@dataclass
class ImpactPrediction:
    """Prediction of market impact."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: int

    # Predicted impacts per venue
    venue_impacts: Dict[Venue, float]  # Venue -> predicted impact bps

    # Best venue
    best_venue: Venue
    best_expected_price: float

    # Confidence
    confidence: float

    # Model info
    model_version: str


class BayesianMarketImpactModel:
    """
    Bayesian model for predicting market impact.

    Transforms SOR from cost-saver to alpha generator.
    Predicts short-term price impact of orders.
    """

    def __init__(self):
        """Initialize the impact model."""
        # Prior beliefs about impact parameters
        # Impact ~ alpha * (quantity / ADV)^beta
        self.alpha_mean = 0.1  # 10% of participation rate
        self.alpha_std = 0.05
        self.beta_mean = 0.5   # Square root impact
        self.beta_std = 0.1

        # Venue-specific adjustments
        self.venue_alpha_adj: Dict[Venue, float] = {
            Venue.NYSE: 1.0,
            Venue.NASDAQ: 0.95,
            Venue.IEX: 0.7,       # Less impact due to speed bump
            Venue.CROSSFINDER: 0.5,  # Dark pool = less impact
            Venue.SIGMA_X: 0.55,
            Venue.LIQUIDNET: 0.4,
        }

        # Historical observations for Bayesian updating
        self.observations: List[Dict[str, Any]] = []

        # Symbol-specific learned parameters
        self.symbol_params: Dict[str, Dict[str, float]] = {}

        self._lock = threading.Lock()

        logger.info("[IMPACT] Bayesian Market Impact Model initialized")

    def predict_impact(
        self,
        symbol: str,
        side: str,
        quantity: int,
        current_price: float,
        adv: float,  # Average daily volume (shares)
        spread_bps: float = 1.0,
        volatility: float = 0.02
    ) -> ImpactPrediction:
        """
        Predict market impact for an order.

        Uses Bayesian inference with learned parameters.
        """
        # Participation rate
        participation = quantity / adv if adv > 0 else 0.01

        # Get symbol-specific parameters or use priors
        params = self.symbol_params.get(symbol, {
            "alpha": self.alpha_mean,
            "beta": self.beta_mean
        })

        alpha = params["alpha"]
        beta = params["beta"]

        # Base impact (percentage)
        base_impact = alpha * (participation ** beta)

        # Volatility adjustment
        vol_adj = 1 + (volatility - 0.02) * 10  # Higher vol = more impact

        # Spread adjustment
        spread_adj = 1 + (spread_bps - 1) * 0.1

        # Calculate impact per venue
        venue_impacts = {}
        for venue, adj in self.venue_alpha_adj.items():
            # Adjusted impact
            impact_bps = base_impact * vol_adj * spread_adj * adj * 10000
            venue_impacts[venue] = impact_bps

        # Find best venue (lowest impact)
        best_venue = min(venue_impacts.items(), key=lambda x: x[1])[0]
        best_impact = venue_impacts[best_venue]

        # Calculate expected price after impact
        if side == "BUY":
            best_expected_price = current_price * (1 + best_impact / 10000)
        else:
            best_expected_price = current_price * (1 - best_impact / 10000)

        # Confidence based on observation count
        n_obs = len([o for o in self.observations if o.get("symbol") == symbol])
        confidence = min(0.95, 0.5 + n_obs * 0.05)

        return ImpactPrediction(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            venue_impacts=venue_impacts,
            best_venue=best_venue,
            best_expected_price=best_expected_price,
            confidence=confidence,
            model_version="bayesian_v1"
        )

    def update_from_execution(
        self,
        symbol: str,
        quantity: int,
        adv: float,
        predicted_impact_bps: float,
        actual_impact_bps: float,
        venue: Venue
    ):
        """
        Update model from observed execution.

        Uses Bayesian updating to refine parameters.
        """
        with self._lock:
            # Record observation
            self.observations.append({
                "symbol": symbol,
                "quantity": quantity,
                "adv": adv,
                "predicted": predicted_impact_bps,
                "actual": actual_impact_bps,
                "venue": venue,
                "timestamp": datetime.utcnow()
            })

            # Keep last 1000 observations
            if len(self.observations) > 1000:
                self.observations = self.observations[-1000:]

            # Update symbol-specific parameters
            symbol_obs = [o for o in self.observations if o["symbol"] == symbol]

            if len(symbol_obs) >= 5:
                # Simple Bayesian update
                # Posterior = weighted average of prior and data
                errors = [(o["actual"] - o["predicted"]) / o["predicted"]
                         for o in symbol_obs if o["predicted"] > 0]

                if errors:
                    avg_error = np.mean(errors)

                    # Adjust alpha based on systematic over/under prediction
                    current_alpha = self.symbol_params.get(
                        symbol, {"alpha": self.alpha_mean}
                    )["alpha"]

                    # Learning rate decreases with more data
                    lr = 0.2 / np.sqrt(len(symbol_obs))
                    new_alpha = current_alpha * (1 + avg_error * lr)
                    new_alpha = max(0.01, min(0.5, new_alpha))  # Bound

                    self.symbol_params[symbol] = {
                        "alpha": new_alpha,
                        "beta": self.beta_mean  # Keep beta fixed for now
                    }

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        with self._lock:
            return {
                "total_observations": len(self.observations),
                "symbols_learned": len(self.symbol_params),
                "avg_prediction_error": np.mean([
                    abs(o["actual"] - o["predicted"])
                    for o in self.observations
                ]) if self.observations else 0
            }


class SmartOrderRouter:
    """
    Smart Order Router - optimizes order routing.

    Goals:
    1. Minimize execution costs
    2. Capture rebates where possible
    3. Reduce information leakage
    4. Get best price improvement
    5. Predict and minimize market impact (NEW)
    """

    def __init__(self):
        """Initialize the SOR."""
        self.venues = VENUE_STATS.copy()
        self.execution_history: List[ExecutionReport] = []
        self._lock = threading.Lock()

        # Learned preferences (would be ML-based in production)
        self.venue_preferences: Dict[str, Dict[Venue, float]] = {}

        logger.info(
            f"[SOR] Smart Order Router initialized with "
            f"{len(self.venues)} venues"
        )

    def route_order(
        self,
        symbol: str,
        side: str,  # BUY, SELL
        quantity: int,
        current_price: float,
        urgency: OrderUrgency = OrderUrgency.NEUTRAL,
        min_fill_rate: float = 0.0,
        prefer_rebates: bool = False
    ) -> RoutingDecision:
        """
        Route an order across venues.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            current_price: Current market price
            urgency: How fast we need to fill
            min_fill_rate: Minimum acceptable fill rate
            prefer_rebates: Prioritize maker rebates
        """
        # Order value
        order_value = quantity * current_price

        # Score each venue
        venue_scores = []

        for venue, stats in self.venues.items():
            # Check minimum size for dark pools
            if stats.min_order_size > quantity:
                continue

            # Calculate score
            score = self._score_venue(
                stats, quantity, urgency, prefer_rebates, order_value
            )

            if stats.fill_rate >= min_fill_rate:
                venue_scores.append((venue, score, stats))

        if not venue_scores:
            # Fallback to primary exchange
            return self._fallback_routing(symbol, side, quantity, current_price)

        # Sort by score (higher is better)
        venue_scores.sort(key=lambda x: x[1], reverse=True)

        # Allocate across venues
        primary_venue, primary_score, primary_stats = venue_scores[0]

        # For large orders, split across venues
        if quantity > 1000 and len(venue_scores) > 1:
            allocations = self._split_order(quantity, venue_scores)
            primary_qty = allocations[0][1]
            secondary = allocations[1:]
        else:
            primary_qty = quantity
            secondary = []

        # Calculate expected metrics
        expected_fill = self._estimate_fill_rate(primary_stats, secondary, quantity)
        expected_cost = self._estimate_cost(primary_stats, secondary, order_value)
        expected_time = primary_stats.avg_fill_time_ms

        # Is limit order
        is_limit = urgency != OrderUrgency.AGGRESSIVE
        limit_price = None
        if is_limit:
            # Slight improvement for limit order
            if side == "BUY":
                limit_price = current_price * 0.9995  # 5 bps below
            else:
                limit_price = current_price * 1.0005  # 5 bps above

        # Routing reason
        if primary_stats.is_dark:
            reason = f"Dark pool {primary_venue.value} for price improvement"
        elif prefer_rebates:
            reason = f"Lit venue {primary_venue.value} for maker rebates"
        elif urgency == OrderUrgency.AGGRESSIVE:
            reason = f"Fast fill at {primary_venue.value}"
        else:
            reason = f"Optimal routing to {primary_venue.value}"

        return RoutingDecision(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            primary_venue=primary_venue,
            primary_quantity=primary_qty,
            secondary_venues=secondary,
            expected_fill_rate=expected_fill,
            expected_cost_bps=expected_cost * 10000,
            expected_fill_time_ms=expected_time,
            is_limit=is_limit,
            limit_price=limit_price,
            routing_reason=reason
        )

    def _score_venue(
        self,
        stats: VenueStats,
        quantity: int,
        urgency: OrderUrgency,
        prefer_rebates: bool,
        order_value: float
    ) -> float:
        """Score a venue for this order."""
        score = 0.0

        # Fill rate (0-30 points)
        score += stats.fill_rate * 30

        # Cost efficiency
        if prefer_rebates and stats.maker_rebate > 0:
            score += stats.maker_rebate * 10000  # Reward rebates
        else:
            score -= stats.taker_fee * 5000  # Penalize fees

        # Price improvement (0-20 points)
        score += stats.price_improvement_bps * 4

        # Information leakage penalty
        score -= stats.information_leakage * 15

        # Urgency adjustments
        if urgency == OrderUrgency.AGGRESSIVE:
            # Prioritize fill rate and speed
            score += stats.fill_rate * 20
            score -= stats.avg_fill_time_ms / 50
        elif urgency == OrderUrgency.PASSIVE:
            # Prioritize improvement and low leakage
            score += stats.price_improvement_bps * 6
            score -= stats.information_leakage * 10

        # Dark pool bonus for large orders
        if stats.is_dark and quantity > 500:
            score += 10

        return score

    def _split_order(
        self,
        quantity: int,
        venue_scores: List[Tuple[Venue, float, VenueStats]]
    ) -> List[Tuple[Venue, int]]:
        """Split order across multiple venues."""
        allocations = []
        remaining = quantity

        # Primary gets 60%
        primary_qty = int(quantity * 0.6)
        allocations.append((venue_scores[0][0], primary_qty))
        remaining -= primary_qty

        # Secondary venues get the rest
        for venue, score, stats in venue_scores[1:4]:  # Up to 3 secondary
            if remaining <= 0:
                break

            alloc = min(remaining, int(quantity * 0.15))
            if alloc >= stats.min_order_size:
                allocations.append((venue, alloc))
                remaining -= alloc

        # Any remaining to primary
        if remaining > 0:
            allocations[0] = (allocations[0][0], allocations[0][1] + remaining)

        return allocations

    def _estimate_fill_rate(
        self,
        primary: VenueStats,
        secondary: List[Tuple[Venue, int]],
        total_qty: int
    ) -> float:
        """Estimate overall fill rate."""
        if not secondary:
            return primary.fill_rate

        total_expected = primary.fill_rate * (total_qty - sum(s[1] for s in secondary))

        for venue, qty in secondary:
            if venue in self.venues:
                total_expected += self.venues[venue].fill_rate * qty

        return min(1.0, total_expected / total_qty)

    def _estimate_cost(
        self,
        primary: VenueStats,
        secondary: List[Tuple[Venue, int]],
        order_value: float
    ) -> float:
        """Estimate total cost as fraction of order value."""
        # Primary cost
        cost = primary.taker_fee - primary.maker_rebate

        # Subtract price improvement
        cost -= primary.price_improvement_bps / 10000

        return cost

    def _fallback_routing(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float
    ) -> RoutingDecision:
        """Fallback routing to primary exchange."""
        return RoutingDecision(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            primary_venue=Venue.NYSE,
            primary_quantity=quantity,
            secondary_venues=[],
            expected_fill_rate=0.85,
            expected_cost_bps=3.0,
            expected_fill_time_ms=50,
            is_limit=False,
            limit_price=None,
            routing_reason="Fallback to primary exchange"
        )

    def simulate_execution(
        self,
        decision: RoutingDecision,
        actual_price: float
    ) -> ExecutionReport:
        """Simulate execution of a routing decision."""
        fills = []
        total_filled = 0
        total_value = 0
        total_fees = 0
        total_rebates = 0

        # Primary venue fill
        primary_stats = self.venues.get(decision.primary_venue)
        if primary_stats:
            fill_qty = int(decision.primary_quantity * primary_stats.fill_rate * np.random.uniform(0.9, 1.1))
            fill_qty = min(fill_qty, decision.primary_quantity)

            # Price with improvement
            improvement = primary_stats.price_improvement_bps / 10000
            if decision.side == "BUY":
                fill_price = actual_price * (1 - improvement)
            else:
                fill_price = actual_price * (1 + improvement)

            fills.append({
                "venue": decision.primary_venue.value,
                "quantity": fill_qty,
                "price": fill_price
            })

            total_filled += fill_qty
            total_value += fill_qty * fill_price

            # Fees
            fill_value = fill_qty * fill_price
            if decision.is_limit:
                total_rebates += fill_value * primary_stats.maker_rebate
            else:
                total_fees += fill_value * primary_stats.taker_fee

        # Secondary venues
        for venue, qty in decision.secondary_venues:
            stats = self.venues.get(venue)
            if not stats:
                continue

            fill_qty = int(qty * stats.fill_rate * np.random.uniform(0.8, 1.0))
            fill_qty = min(fill_qty, qty)

            improvement = stats.price_improvement_bps / 10000
            if decision.side == "BUY":
                fill_price = actual_price * (1 - improvement)
            else:
                fill_price = actual_price * (1 + improvement)

            fills.append({
                "venue": venue.value,
                "quantity": fill_qty,
                "price": fill_price
            })

            total_filled += fill_qty
            total_value += fill_qty * fill_price
            total_fees += fill_qty * fill_price * stats.taker_fee

        # Calculate metrics
        avg_fill_price = total_value / total_filled if total_filled > 0 else actual_price

        # Implementation shortfall
        if decision.side == "BUY":
            shortfall = (avg_fill_price - actual_price) / actual_price
        else:
            shortfall = (actual_price - avg_fill_price) / actual_price

        # Price improvement (negative shortfall = improvement)
        price_improvement = -shortfall * 10000  # Convert to bps

        report = ExecutionReport(
            timestamp=datetime.utcnow(),
            symbol=decision.symbol,
            side=decision.side,
            quantity=decision.quantity,
            fills=fills,
            total_filled=total_filled,
            avg_fill_price=avg_fill_price,
            arrival_price=actual_price,
            implementation_shortfall_bps=shortfall * 10000,
            total_fees=total_fees,
            total_rebates=total_rebates,
            net_cost=total_fees - total_rebates,
            price_improvement_bps=price_improvement,
            fill_rate=total_filled / decision.quantity if decision.quantity > 0 else 0,
            avg_fill_time_ms=decision.expected_fill_time_ms
        )

        with self._lock:
            self.execution_history.append(report)

        return report

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution quality statistics."""
        with self._lock:
            if not self.execution_history:
                return {}

            reports = self.execution_history[-100:]  # Last 100

            return {
                "total_executions": len(reports),
                "avg_fill_rate": np.mean([r.fill_rate for r in reports]),
                "avg_price_improvement_bps": np.mean([r.price_improvement_bps for r in reports]),
                "avg_implementation_shortfall_bps": np.mean([r.implementation_shortfall_bps for r in reports]),
                "total_fees": sum(r.total_fees for r in reports),
                "total_rebates": sum(r.total_rebates for r in reports),
                "net_cost": sum(r.net_cost for r in reports)
            }


# Singleton
_sor: Optional[SmartOrderRouter] = None


def get_smart_order_router() -> SmartOrderRouter:
    """Get or create the Smart Order Router."""
    global _sor
    if _sor is None:
        _sor = SmartOrderRouter()
    return _sor
