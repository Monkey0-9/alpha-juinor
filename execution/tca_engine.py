"""
Transaction Cost Analysis (TCA) Engine
======================================

Comprehensive TCA for institutional execution quality analysis.

Features:
- Pre-trade cost estimation
- Post-trade performance attribution
- Slippage decomposition (timing, impact, spread)
- Execution quality scorecards
- Automated feedback to alpha models

Phase 2.4: Transaction Cost Analysis
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Metrics for a single execution."""
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    arrival_price: float
    exec_price: float
    vwap_price: float
    close_price: float
    exec_time_seconds: float


@dataclass
class TCAReport:
    """Comprehensive TCA report."""
    # Slippage Components (in basis points)
    total_slippage_bps: float
    timing_cost_bps: float
    impact_cost_bps: float
    spread_cost_bps: float

    # Quality Metrics
    implementation_shortfall_bps: float
    vwap_deviation_bps: float
    participation_rate: float

    # Scores (0-100)
    execution_quality_score: float
    timing_score: float
    venue_score: float

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TCAEngine:
    """
    Transaction Cost Analysis Engine for execution quality.
    """

    def __init__(self):
        self.history: List[TCAReport] = []
        logger.info("TCA Engine initialized")

    def analyze_execution(
        self, metrics: ExecutionMetrics
    ) -> TCAReport:
        """
        Analyze a single execution and generate TCA report.
        """
        # Direction multiplier
        direction = 1.0 if metrics.side == "BUY" else -1.0

        # Calculate slippage components
        arrival = metrics.arrival_price
        exec_p = metrics.exec_price
        vwap = metrics.vwap_price
        close = metrics.close_price

        # Total slippage: (exec - arrival) * direction / arrival
        total_slip = (exec_p - arrival) * direction / arrival * 10000

        # Timing cost: (arrival - close) * direction / arrival
        timing = (close - arrival) * direction / arrival * 10000

        # Impact cost: remaining after timing
        impact = total_slip - timing * 0.3

        # Spread cost estimate (simplified)
        spread = abs(exec_p - vwap) / arrival * 10000 * 0.5

        # Implementation Shortfall
        is_bps = total_slip

        # VWAP deviation
        vwap_dev = (exec_p - vwap) * direction / vwap * 10000

        # Participation rate (simplified estimate)
        part_rate = min(1.0, metrics.quantity / 10000)

        # Scores
        exec_score = max(0, 100 - abs(total_slip))
        timing_score = max(0, 100 - abs(timing))
        venue_score = max(0, 100 - abs(spread) * 2)

        report = TCAReport(
            total_slippage_bps=total_slip,
            timing_cost_bps=timing,
            impact_cost_bps=impact,
            spread_cost_bps=spread,
            implementation_shortfall_bps=is_bps,
            vwap_deviation_bps=vwap_dev,
            participation_rate=part_rate,
            execution_quality_score=exec_score,
            timing_score=timing_score,
            venue_score=venue_score
        )

        self.history.append(report)
        return report

    def get_aggregate_stats(self) -> Dict[str, float]:
        """
        Get aggregate TCA statistics.
        """
        if not self.history:
            return {
                "avg_slippage_bps": 0.0,
                "avg_exec_score": 100.0,
                "total_executions": 0
            }

        slippages = [r.total_slippage_bps for r in self.history]
        scores = [r.execution_quality_score for r in self.history]

        return {
            "avg_slippage_bps": np.mean(slippages),
            "avg_exec_score": np.mean(scores),
            "total_executions": len(self.history),
            "worst_slippage_bps": max(slippages),
            "best_slippage_bps": min(slippages)
        }

    def generate_feedback_for_alpha(self) -> Dict[str, Any]:
        """
        Generate feedback signals for alpha models.

        This implements the closed-loop feedback from execution
        back to alpha/signal generation.
        """
        stats = self.get_aggregate_stats()

        # If average slippage is high, signal alpha to reduce turnover
        if stats["avg_slippage_bps"] > 10:
            return {
                "action": "REDUCE_TURNOVER",
                "severity": "HIGH",
                "reason": f"Avg slippage {stats['avg_slippage_bps']:.1f} bps > 10 bps target"
            }
        elif stats["avg_slippage_bps"] > 5:
            return {
                "action": "REDUCE_TURNOVER",
                "severity": "MEDIUM",
                "reason": f"Avg slippage {stats['avg_slippage_bps']:.1f} bps > 5 bps warning"
            }
        else:
            return {
                "action": "NONE",
                "severity": "LOW",
                "reason": "Execution quality within targets"
            }


# Singleton
_tca_engine = None


def get_tca_engine() -> TCAEngine:
    global _tca_engine
    if _tca_engine is None:
        _tca_engine = TCAEngine()
    return _tca_engine
