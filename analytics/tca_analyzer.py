"""
Execution Quality Metrics - TCA (Transaction Cost Analysis).

Features:
- Implementation shortfall
- VWAP comparison
- Slippage analysis
- Market impact estimation
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Execution quality metrics for a single order."""
    order_id: str
    symbol: str
    side: str

    # Prices
    decision_price: float
    arrival_price: float
    execution_price: float
    close_price: float
    vwap: float

    # Quantities
    target_quantity: int
    filled_quantity: int

    # Metrics (in bps)
    implementation_shortfall: float
    arrival_cost: float
    execution_cost: float
    market_impact: float
    slippage: float
    timing_cost: float

    # Comparison
    vs_vwap: float
    vs_twap: float

    # Quality score (0-100)
    quality_score: float


@dataclass
class AggregateTCA:
    """Aggregate TCA report."""
    period: str
    total_orders: int
    total_volume: float

    avg_shortfall: float
    avg_slippage: float
    avg_impact: float

    cost_saved_vs_vwap: float
    quality_score: float

    by_algorithm: Dict[str, Dict[str, float]]
    by_venue: Dict[str, Dict[str, float]]


class ExecutionQualityAnalyzer:
    """
    Transaction Cost Analysis (TCA).

    Measures:
    - Implementation Shortfall
    - Arrival Cost
    - Market Impact
    - Timing Cost
    - VWAP/TWAP comparison
    """

    def __init__(self):
        self.execution_history: List[ExecutionMetrics] = []

        # Venue quality tracking
        self.venue_stats: Dict[str, List[float]] = {}

        # Algo quality tracking
        self.algo_stats: Dict[str, List[float]] = {}

    def analyze_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        decision_price: float,
        arrival_price: float,
        execution_price: float,
        close_price: float,
        vwap: float,
        target_quantity: int,
        filled_quantity: int,
        algo_used: str = "MARKET",
        venue: str = "PRIMARY"
    ) -> ExecutionMetrics:
        """
        Analyze a single execution.
        """
        direction = 1 if side == "BUY" else -1

        # Implementation Shortfall (IS)
        # Paper return - Actual return
        paper_value = target_quantity * decision_price
        actual_value = filled_quantity * execution_price
        unfilled_value = (target_quantity - filled_quantity) * close_price

        is_bps = direction * (
            (actual_value + unfilled_value - paper_value) / paper_value
        ) * 10000

        # Arrival Cost
        arrival_cost = direction * (
            (execution_price - arrival_price) / arrival_price
        ) * 10000

        # Execution Cost (vs decision)
        exec_cost = direction * (
            (execution_price - decision_price) / decision_price
        ) * 10000

        # Market Impact (permanent price impact)
        impact = direction * (
            (close_price - arrival_price) / arrival_price
        ) * 10000

        # Slippage
        slippage = direction * (
            (execution_price - decision_price) / decision_price
        ) * 10000

        # Timing Cost
        timing = direction * (
            (arrival_price - decision_price) / decision_price
        ) * 10000

        # VWAP comparison
        vs_vwap = direction * (
            (vwap - execution_price) / vwap
        ) * 10000

        # TWAP (simplified as average of arrival and close)
        twap = (arrival_price + close_price) / 2
        vs_twap = direction * (
            (twap - execution_price) / twap
        ) * 10000

        # Quality score (0-100)
        # Lower slippage and better VWAP = higher score
        quality = 100
        quality -= abs(slippage) * 2  # Penalty for slippage
        quality += vs_vwap * 0.5  # Bonus for beating VWAP
        quality -= abs(impact) * 1  # Penalty for impact
        quality = max(0, min(100, quality))

        metrics = ExecutionMetrics(
            order_id=order_id,
            symbol=symbol,
            side=side,
            decision_price=decision_price,
            arrival_price=arrival_price,
            execution_price=execution_price,
            close_price=close_price,
            vwap=vwap,
            target_quantity=target_quantity,
            filled_quantity=filled_quantity,
            implementation_shortfall=is_bps,
            arrival_cost=arrival_cost,
            execution_cost=exec_cost,
            market_impact=impact,
            slippage=slippage,
            timing_cost=timing,
            vs_vwap=vs_vwap,
            vs_twap=vs_twap,
            quality_score=quality
        )

        self.execution_history.append(metrics)

        # Track by venue
        if venue not in self.venue_stats:
            self.venue_stats[venue] = []
        self.venue_stats[venue].append(slippage)

        # Track by algo
        if algo_used not in self.algo_stats:
            self.algo_stats[algo_used] = []
        self.algo_stats[algo_used].append(slippage)

        return metrics

    def get_aggregate_tca(self, period: str = "daily") -> AggregateTCA:
        """Get aggregate TCA report."""
        if not self.execution_history:
            return AggregateTCA(
                period=period,
                total_orders=0,
                total_volume=0,
                avg_shortfall=0,
                avg_slippage=0,
                avg_impact=0,
                cost_saved_vs_vwap=0,
                quality_score=0,
                by_algorithm={},
                by_venue={}
            )

        # Aggregate metrics
        n = len(self.execution_history)
        total_volume = sum(
            m.filled_quantity * m.execution_price
            for m in self.execution_history
        )

        avg_shortfall = np.mean([m.implementation_shortfall for m in self.execution_history])
        avg_slippage = np.mean([m.slippage for m in self.execution_history])
        avg_impact = np.mean([m.market_impact for m in self.execution_history])
        cost_saved = np.mean([m.vs_vwap for m in self.execution_history])
        quality = np.mean([m.quality_score for m in self.execution_history])

        # By venue
        by_venue = {}
        for venue, slippages in self.venue_stats.items():
            by_venue[venue] = {
                "avg_slippage": np.mean(slippages),
                "order_count": len(slippages)
            }

        # By algo
        by_algo = {}
        for algo, slippages in self.algo_stats.items():
            by_algo[algo] = {
                "avg_slippage": np.mean(slippages),
                "order_count": len(slippages)
            }

        return AggregateTCA(
            period=period,
            total_orders=n,
            total_volume=total_volume,
            avg_shortfall=avg_shortfall,
            avg_slippage=avg_slippage,
            avg_impact=avg_impact,
            cost_saved_vs_vwap=cost_saved,
            quality_score=quality,
            by_algorithm=by_algo,
            by_venue=by_venue
        )

    def get_best_algo(self) -> str:
        """Get best performing algorithm."""
        if not self.algo_stats:
            return "VWAP"

        best = min(
            self.algo_stats.items(),
            key=lambda x: np.mean(x[1])
        )
        return best[0]

    def get_best_venue(self) -> str:
        """Get best performing venue."""
        if not self.venue_stats:
            return "PRIMARY"

        best = min(
            self.venue_stats.items(),
            key=lambda x: np.mean(x[1])
        )
        return best[0]


# Global singleton
_tca: Optional[ExecutionQualityAnalyzer] = None


def get_tca_analyzer() -> ExecutionQualityAnalyzer:
    """Get or create global TCA analyzer."""
    global _tca
    if _tca is None:
        _tca = ExecutionQualityAnalyzer()
    return _tca
