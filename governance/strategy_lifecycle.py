"""
Strategy Lifecycle Manager - Phase 10
======================================
Manages strategy lifecycle stages: INCUBATING → SCALING → HARVESTING → DECOMMISSIONED

Capital decays by default. Strategies must prove continued effectiveness.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from governance.institutional_specification import (
    StrategyLifecycle,
    StrategyLifecycleState
)

logger = logging.getLogger(__name__)


class StrategyLifecycleManager:
    """
    Manages strategy lifecycle stages and capital allocation limits.

    Each strategy must be in one of four stages:
    - INCUBATING: New strategy, max 2% NAV
    - SCALING: Proven strategy, max 5% NAV
    - HARVESTING: Mature strategy, max 10% NAV
    - DECOMMISSIONED: Failed/retired strategy, 0% NAV

    Capital decays by default - strategies must prove continued effectiveness.
    """

    def __init__(
        self,
        default_max_capital_pct: Dict[StrategyLifecycle, float] = None,
        decay_rate: float = 0.95,
        # Stage transition thresholds
        incubating_to_scaling_sharpe: float = 0.8,
        scaling_to_harvesting_sharpe: float = 1.2,
        decommission_sharpe: float = 0.3,
        decommission_max_drawdown: float = 0.15,
        decommission_profit_factor: float = 0.8,
        decommission_error_rate: float = 0.05,
        # Performance lookback
        performance_window_days: int = 21,
        min_profit_count: int = 5
    ):
        """
        Initialize Strategy Lifecycle Manager.

        Args:
            default_max_capital_pct: Capital limits per stage
            decay_rate: Daily decay rate for capital allocation
            incubating_to_scaling_sharpe: Sharpe threshold to scale up
            scaling_to_harvesting_sharpe: Sharpe threshold to harvest
            decommission_sharpe: Sharpe below which to decommission
            decommission_max_drawdown: DD above which to decommission
            decommission_profit_factor: PF below which to decommission
            decommission_error_rate: Error rate above which to decommission
            performance_window_days: Days for performance lookback
            min_profit_count: Minimum profitable signals for assessment
        """
        self.default_max_capital_pct = default_max_capital_pct or {
            StrategyLifecycle.INCUBATING: 0.02,
            StrategyLifecycle.SCALING: 0.05,
            StrategyLifecycle.HARVESTING: 0.10,
            StrategyLifecycle.DECOMMISSIONED: 0.0
        }
        self.decay_rate = decay_rate
        self.incubating_to_scaling_sharpe = incubating_to_scaling_sharpe
        self.scaling_to_harvesting_sharpe = scaling_to_harvesting_sharpe
        self.decommission_sharpe = decommission_sharpe
        self.decommission_max_drawdown = decommission_max_drawdown
        self.decommission_profit_factor = decommission_profit_factor
        self.decommission_error_rate = decommission_error_rate
        self.performance_window_days = performance_window_days
        self.min_profit_count = min_profit_count

        # Track strategy states
        self._strategy_states: Dict[str, StrategyLifecycleState] = {}

        # Performance history (symbol -> list of (date, return))
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}

    def register_strategy(
        self,
        strategy_id: str,
        initial_stage: StrategyLifecycle = StrategyLifecycle.INCUBATING,
        metadata: Dict[str, Any] = None
    ) -> StrategyLifecycleState:
        """
        Register a new strategy.

        Args:
            strategy_id: Unique strategy identifier
            initial_stage: Starting lifecycle stage
            metadata: Additional strategy metadata

        Returns:
            StrategyLifecycleState
        """
        state = StrategyLifecycleState(
            strategy_id=strategy_id,
            stage=initial_stage,
            max_capital_pct=self.default_max_capital_pct.copy(),
            decay_rate=self.decay_rate
        )

        self._strategy_states[strategy_id] = state
        self._performance_history[strategy_id] = []

        logger.info(f"[LIFECYCLE] Registered strategy {strategy_id} at stage {initial_stage.value}")

        return state

    def update_performance(
        self,
        strategy_id: str,
        return_pct: float,
        timestamp: datetime = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Update strategy performance history.

        Args:
            strategy_id: Strategy identifier
            return_pct: Return for the period
            timestamp: Timestamp of the update
            metadata: Additional performance metadata
        """
        if strategy_id not in self._strategy_states:
            self.register_strategy(strategy_id)

        if timestamp is None:
            timestamp = datetime.utcnow()

        self._performance_history[strategy_id].append({
            'timestamp': timestamp.isoformat(),
            'return_pct': return_pct,
            'metadata': metadata or {}
        })

        # Prune old history
        cutoff = datetime.utcnow() - timedelta(days=self.performance_window_days)
        self._performance_history[strategy_id] = [
            p for p in self._performance_history[strategy_id]
            if datetime.fromisoformat(p['timestamp']) > cutoff
        ]

    def compute_strategy_metrics(
        self,
        strategy_id: str
    ) -> Dict[str, Any]:
        """
        Compute performance metrics for a strategy.

        Returns:
            Dict with total_return, sharpe_rolling, max_drawdown, profit_factor, error_rate
        """
        if strategy_id not in self._performance_history:
            return {
                'total_return': 0.0,
                'sharpe_rolling': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'error_rate': 0.0,
                'profit_count': 0,
                'loss_count': 0,
                'signal_count': 0
            }

        history = self._performance_history[strategy_id]
        if not history:
            return {
                'total_return': 0.0,
                'sharpe_rolling': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'error_rate': 0.0,
                'profit_count': 0,
                'loss_count': 0,
                'signal_count': 0
            }

        returns = [p['return_pct'] for p in history]
        signal_count = len(returns)

        # Total return
        total_return = sum(returns)

        # Sharpe ratio (annualized)
        if len(returns) > 1:
            mean_ret = sum(returns) / len(returns)
            std_ret = (sum((r - mean_ret) ** 2 for r in returns) / len(returns)) ** 0.5
            if std_ret > 0:
                # Annualize: daily to annual
                sharpe = (mean_ret / std_ret) * (252 ** 0.5)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in returns:
            cumulative += r
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / (peak + 1e-10) if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Profit factor
        profits = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))
        profit_factor = profits / (losses + 1e-10)

        # Error rate (from metadata)
        error_count = sum(
            1 for p in history
            if p.get('metadata', {}).get('error', False)
        )
        error_rate = error_count / signal_count if signal_count > 0 else 0.0

        profit_count = sum(1 for r in returns if r > 0)
        loss_count = sum(1 for r in returns if r < 0)

        return {
            'total_return': total_return,
            'sharpe_rolling': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'error_rate': error_rate,
            'profit_count': profit_count,
            'loss_count': loss_count,
            'signal_count': signal_count
        }

    def evaluate_stage_transition(
        self,
        strategy_id: str
    ) -> Optional[StrategyLifecycle]:
        """
        Evaluate if strategy should transition to a new stage.

        Returns:
            New stage if transition needed, None otherwise
        """
        if strategy_id not in self._strategy_states:
            return None

        state = self._strategy_states[strategy_id]
        metrics = self.compute_strategy_metrics(strategy_id)

        # Update state metrics
        state.total_return = metrics['total_return']
        state.sharpe_rolling = metrics['sharpe_rolling']
        state.max_drawdown = metrics['max_drawdown']
        state.profit_factor = metrics['profit_factor']
        state.error_rate = metrics['error_rate']

        # Check for decommission (original logic)
        if state.should_decommission():
            logger.warning(
                f"[LIFECYCLE] Decommissioning {strategy_id}: "
                f"Sharpe={metrics['sharpe_rolling']:.2f}, "
                f"DD={metrics['max_drawdown']:.1%}, "
                f"PF={metrics['profit_factor']:.2f}"
            )
            return StrategyLifecycle.DECOMMISSIONED

        # NEW: Check for decay-based degradation
        # If strategy has decay metrics attached (from AlphaDecayMonitor)
        decay_status = getattr(state, 'decay_status', None)
        if decay_status == "CRITICAL":
            logger.warning(f"[LIFECYCLE] CRITICAL decay detected for {strategy_id}, moving to DECOMMISSIONED")
            return StrategyLifecycle.DECOMMISSIONED
        elif decay_status == "DEGRADED":
            # Demote to INCUBATING (quarantine)
            if state.stage in [StrategyLifecycle.SCALING, StrategyLifecycle.HARVESTING]:
                logger.warning(f"[LIFECYCLE] DEGRADED decay for {strategy_id}, demoting to INCUBATING")
                return StrategyLifecycle.INCUBATING

        # Check stage-specific transitions
        if state.stage == StrategyLifecycle.INCUBATING:
            # Need consistent profitability to scale
            if (metrics['sharpe_rolling'] >= self.incubating_to_scaling_sharpe and
                metrics['profit_count'] >= self.min_profit_count):
                logger.info(
                    f"[LIFECYCLE] Promoting {strategy_id} INCUBATING → SCALING "
                    f"(Sharpe={metrics['sharpe_rolling']:.2f})"
                )
                return StrategyLifecycle.SCALING

        elif state.stage == StrategyLifecycle.SCALING:
            # Need strong performance to harvest
            if (metrics['sharpe_rolling'] >= self.scaling_to_harvesting_sharpe and
                metrics['profit_count'] >= self.min_profit_count * 2):
                logger.info(
                    f"[LIFECYCLE] Promoting {strategy_id} SCALING → HARVESTING "
                    f"(Sharpe={metrics['sharpe_rolling']:.2f})"
                )
                return StrategyLifecycle.HARVESTING

        elif state.stage == StrategyLifecycle.HARVESTING:
            # Apply decay to encourage rotation
            # Still stays in harvesting unless severely underperforming
            if metrics['sharpe_rolling'] < self.decommission_sharpe * 0.5:
                logger.warning(
                    f"[LIFECYCLE] Demoting {strategy_id} HARVESTING → SCALING "
                    f"(Sharpe={metrics['sharpe_rolling']:.2f})"
                )
                return StrategyLifecycle.SCALING

        return None

    def apply_stage_transition(
        self,
        strategy_id: str,
        new_stage: StrategyLifecycle
    ) -> StrategyLifecycleState:
        """
        Apply a stage transition to a strategy.

        Args:
            strategy_id: Strategy identifier
            new_stage: New lifecycle stage

        Returns:
            Updated StrategyLifecycleState
        """
        if strategy_id not in self._strategy_states:
            self.register_strategy(strategy_id, new_stage)
            return self._strategy_states[strategy_id]

        state = self._strategy_states[strategy_id]
        old_stage = state.stage

        state.stage = new_stage

        logger.info(
            f"[LIFECYCLE] Transitioned {strategy_id}: "
            f"{old_stage.value} → {new_stage.value}"
        )

        return state

    def get_capital_limit(
        self,
        strategy_id: str,
        portfolio_nav: float
    ) -> float:
        """
        Get the capital allocation limit for a strategy.

        Args:
            strategy_id: Strategy identifier
            portfolio_nav: Total portfolio NAV

        Returns:
            Maximum capital allocation in dollars
        """
        if strategy_id not in self._strategy_states:
            # New strategy starts at INCUBATING limits
            self.register_strategy(strategy_id)

        state = self._strategy_states[strategy_id]

        # Evaluate and apply any pending transitions
        new_stage = self.evaluate_stage_transition(strategy_id)
        if new_stage is not None:
            self.apply_stage_transition(strategy_id, new_stage)

        # Get base limit from stage
        base_limit = state.get_current_capital_limit(portfolio_nav)

        # Apply time decay (capital decays by default)
        # Strategies need to prove continued effectiveness
        days_active = getattr(state, 'days_active', 0)
        decay_factor = state.decay_rate ** days_active

        final_limit = base_limit * decay_factor

        return final_limit

    def get_strategy_state(
        self,
        strategy_id: str
    ) -> Optional[StrategyLifecycleState]:
        """Get the current state of a strategy."""
        return self._strategy_states.get(strategy_id)

    def get_all_strategy_states(self) -> Dict[str, StrategyLifecycleState]:
        """Get states of all registered strategies."""
        return self._strategy_states.copy()

    def decommission_strategy(
        self,
        strategy_id: str,
        reason: str = "Manual decommission"
    ) -> bool:
        """
        Immediately decommission a strategy.

        Args:
            strategy_id: Strategy identifier
            reason: Reason for decommissioning

        Returns:
            True if successful, False if strategy not found
        """
        if strategy_id not in self._strategy_states:
            return False

        state = self._strategy_states[strategy_id]
        old_stage = state.stage

        self.apply_stage_transition(strategy_id, StrategyLifecycle.DECOMMISSIONED)

        logger.warning(
            f"[LIFECYCLE] DECOMMISSIONED {strategy_id}: {reason} "
            f"(was {old_stage.value})"
        )

        return True

    def get_lifecycle_summary(self) -> Dict[str, Any]:
        """Get summary of all strategy lifecycles."""
        summary = {
            'total_strategies': len(self._strategy_states),
            'by_stage': {
                stage.value: 0 for stage in StrategyLifecycle
            },
            'total_allocated': 0.0,
            'decommissioned_count': 0,
            'strategies': []
        }

        for strategy_id, state in self._strategy_states.items():
            stage = state.stage.value
            summary['by_stage'][stage] += 1

            if stage == StrategyLifecycle.DECOMMISSIONED.value:
                summary['decommissioned_count'] += 1

            metrics = self.compute_strategy_metrics(strategy_id)

            summary['strategies'].append({
                'strategy_id': strategy_id,
                'stage': stage,
                'sharpe_rolling': metrics['sharpe_rolling'],
                'max_drawdown': metrics['max_drawdown'],
                'profit_factor': metrics['profit_factor'],
                'total_return': metrics['total_return'],
                'signal_count': metrics['signal_count']
            })

        return summary

    def integrate_decay_metrics(self, strategy_id: str, decay_status: str, decay_score: float):
        """
        Integrate decay metrics from AlphaDecayMonitor.

        Args:
            strategy_id: Strategy identifier
            decay_status: HEALTHY, DEGRADED, CRITICAL
            decay_score: 0-1 decay score
        """
        if strategy_id not in self._strategy_states:
            self.register_strategy(strategy_id)

        state = self._strategy_states[strategy_id]
        state.decay_status = decay_status
        state.decay_score = decay_score

        logger.info(f"[LIFECYCLE] Updated decay metrics for {strategy_id}: {decay_status} (score={decay_score:.2f})")



# Convenience function
def get_strategy_capital_limit(
    strategy_id: str,
    portfolio_nav: float,
    strategy_states: Dict[str, StrategyLifecycleState] = None
) -> float:
    """
    Get capital limit for a strategy.

    Args:
        strategy_id: Strategy identifier
        portfolio_nav: Total portfolio NAV
        strategy_states: Existing strategy states (optional)

    Returns:
        Maximum capital allocation
    """
    manager = StrategyLifecycleManager()

    if strategy_states and strategy_id in strategy_states:
        manager._strategy_states = strategy_states

    return manager.get_capital_limit(strategy_id, portfolio_nav)

