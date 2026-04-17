"""
Adaptive Strategy Allocator - Dynamic Strategy Rotation.

Features:
- Strategy performance tracking
- Regime-based allocation
- Momentum-based weighting
- Drawdown-adjusted sizing
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    """Strategy performance statistics."""
    strategy_id: str
    cumulative_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.5
    avg_hold_time: float = 0.0
    recent_return_5d: float = 0.0
    recent_return_20d: float = 0.0
    trade_count: int = 0
    capital_deployed: float = 0.0


@dataclass
class AllocationResult:
    """Strategy allocation result."""
    allocations: Dict[str, float]
    scores: Dict[str, float]
    regime: str
    total_allocated: float


class AdaptiveAllocator:
    """
    Adaptive strategy allocation manager.

    Features:
    - Score strategies on risk-adjusted returns
    - Rotate allocation based on regime
    - Momentum-based weighting
    - Maximum concentration limits
    """

    def __init__(
        self,
        max_per_strategy: float = 0.40,
        min_per_strategy: float = 0.05,
        momentum_lookback: int = 20,
        rebalance_threshold: float = 0.10
    ):
        self.max_per_strategy = max_per_strategy
        self.min_per_strategy = min_per_strategy
        self.momentum_lookback = momentum_lookback
        self.rebalance_threshold = rebalance_threshold

        self.strategies: Dict[str, StrategyStats] = {}
        self.current_allocations: Dict[str, float] = {}
        self.return_history: Dict[str, List[float]] = {}

        # Regime-specific weights
        self.regime_preferences = {
            "BULL_LOW_VOL": {
                "momentum": 0.30,
                "mean_reversion": 0.15,
                "trend_following": 0.25,
                "stat_arb": 0.20,
                "event_driven": 0.10
            },
            "BULL_HIGH_VOL": {
                "momentum": 0.10,
                "mean_reversion": 0.25,
                "trend_following": 0.20,
                "stat_arb": 0.25,
                "event_driven": 0.20
            },
            "BEAR_LOW_VOL": {
                "momentum": 0.15,
                "mean_reversion": 0.25,
                "trend_following": 0.15,
                "stat_arb": 0.30,
                "event_driven": 0.15
            },
            "BEAR_HIGH_VOL": {
                "momentum": 0.05,
                "mean_reversion": 0.30,
                "trend_following": 0.10,
                "stat_arb": 0.35,
                "event_driven": 0.20
            },
            "CRISIS": {
                "momentum": 0.05,
                "mean_reversion": 0.35,
                "trend_following": 0.05,
                "stat_arb": 0.40,
                "event_driven": 0.15
            }
        }

    def register_strategy(
        self,
        strategy_id: str,
        strategy_type: str
    ):
        """Register a new strategy."""
        self.strategies[strategy_id] = StrategyStats(strategy_id=strategy_id)
        self.return_history[strategy_id] = []
        logger.info(f"Registered strategy: {strategy_id} ({strategy_type})")

    def update_performance(
        self,
        strategy_id: str,
        daily_return: float,
        trade_count: int = 0
    ):
        """Update strategy performance."""
        if strategy_id not in self.strategies:
            return

        stats = self.strategies[strategy_id]

        # Update cumulative return
        stats.cumulative_return = (
            (1 + stats.cumulative_return) * (1 + daily_return) - 1
        )

        # Store return history
        self.return_history[strategy_id].append(daily_return)
        if len(self.return_history[strategy_id]) > 252:
            self.return_history[strategy_id] = self.return_history[strategy_id][-252:]

        # Update recent returns
        history = self.return_history[strategy_id]
        if len(history) >= 5:
            stats.recent_return_5d = sum(history[-5:])
        if len(history) >= 20:
            stats.recent_return_20d = sum(history[-20:])

        # Update Sharpe ratio
        if len(history) >= 20:
            mean_ret = np.mean(history)
            std_ret = np.std(history)
            stats.sharpe_ratio = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

        # Update drawdown
        cumulative = np.cumsum(history)
        peak = np.maximum.accumulate(cumulative)
        dd = peak - cumulative
        stats.max_drawdown = float(np.max(dd)) if len(dd) > 0 else 0

        stats.trade_count += trade_count

    def score_strategy(
        self,
        strategy_id: str,
        regime: str
    ) -> float:
        """Score strategy for allocation."""
        if strategy_id not in self.strategies:
            return 0.0

        stats = self.strategies[strategy_id]

        # Base score from Sharpe
        sharpe_score = stats.sharpe_ratio * 20

        # Momentum bonus
        momentum_score = stats.recent_return_5d * 100 + stats.recent_return_20d * 50

        # Drawdown penalty
        dd_penalty = stats.max_drawdown * 50

        # Combine
        score = sharpe_score + momentum_score - dd_penalty

        return max(0, score)

    def allocate(
        self,
        regime: str,
        total_capital: float,
        strategy_types: Dict[str, str] = None
    ) -> AllocationResult:
        """
        Allocate capital across strategies.
        """
        scores = {}
        raw_allocations = {}

        # Score each strategy
        for strategy_id in self.strategies:
            score = self.score_strategy(strategy_id, regime)
            scores[strategy_id] = score

        # Normalize to allocations
        total_score = sum(scores.values())

        if total_score == 0:
            # Equal weight if no scores
            for strategy_id in self.strategies:
                raw_allocations[strategy_id] = 1.0 / len(self.strategies)
        else:
            for strategy_id, score in scores.items():
                raw_allocations[strategy_id] = score / total_score

        # Apply limits
        final_allocations = {}
        for strategy_id, alloc in raw_allocations.items():
            final_allocations[strategy_id] = max(
                self.min_per_strategy,
                min(self.max_per_strategy, alloc)
            )

        # Renormalize
        total = sum(final_allocations.values())
        for strategy_id in final_allocations:
            final_allocations[strategy_id] /= total

        # Check if rebalance needed
        if self.current_allocations:
            max_drift = max(
                abs(final_allocations.get(s, 0) - self.current_allocations.get(s, 0))
                for s in set(final_allocations) | set(self.current_allocations)
            )
            if max_drift < self.rebalance_threshold:
                # Use current allocations
                final_allocations = self.current_allocations.copy()

        self.current_allocations = final_allocations

        return AllocationResult(
            allocations=final_allocations,
            scores=scores,
            regime=regime,
            total_allocated=sum(final_allocations.values())
        )


# Global singleton
_allocator: Optional[AdaptiveAllocator] = None


def get_adaptive_allocator() -> AdaptiveAllocator:
    """Get or create global adaptive allocator."""
    global _allocator
    if _allocator is None:
        _allocator = AdaptiveAllocator()
    return _allocator
