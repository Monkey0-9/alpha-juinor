"""
Pod Risk Manager - Millennium/Point72 Multi-Manager Structure.

Features:
- Independent pod P&L tracking
- Automatic drawdown cuts (5% -> 50% cut, 7.5% -> termination)
- Dynamic capital allocation
- Pod performance ranking
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PodStatus(Enum):
    """Pod operating status."""
    ACTIVE = "ACTIVE"
    REDUCED = "REDUCED"  # Capital reduced due to drawdown
    SUSPENDED = "SUSPENDED"  # Temporarily suspended
    TERMINATED = "TERMINATED"  # Permanently shut down


@dataclass
class PodMetrics:
    """Pod performance metrics."""
    pod_id: str
    strategy_name: str

    # Capital
    initial_capital: float
    current_capital: float
    allocated_capital: float

    # Performance
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    mtd_pnl: float = 0.0
    ytd_pnl: float = 0.0

    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    # Status
    status: PodStatus = PodStatus.ACTIVE
    reduction_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class PodAction:
    """Action to take on a pod."""
    pod_id: str
    action: str  # "MAINTAIN", "REDUCE", "TERMINATE", "INCREASE"
    reason: str
    capital_multiplier: float = 1.0


class PodRiskManager:
    """
    Millennium/Point72-style pod risk management.

    Risk Rules:
    - 5% drawdown: Cut capital by 50%
    - 7.5% drawdown: Terminate pod
    - Top performers get capital increases
    - Monthly reallocation
    """

    def __init__(
        self,
        drawdown_cut_threshold: float = 0.05,
        drawdown_cut_amount: float = 0.50,
        termination_threshold: float = 0.075,
        top_performer_bonus: float = 0.20,
        min_sharpe_for_bonus: float = 1.5
    ):
        self.drawdown_cut_threshold = drawdown_cut_threshold
        self.drawdown_cut_amount = drawdown_cut_amount
        self.termination_threshold = termination_threshold
        self.top_performer_bonus = top_performer_bonus
        self.min_sharpe_for_bonus = min_sharpe_for_bonus

        self.pods: Dict[str, PodMetrics] = {}
        self.historical_actions: List[PodAction] = []

    def register_pod(
        self,
        pod_id: str,
        strategy_name: str,
        initial_capital: float
    ) -> PodMetrics:
        """Register a new pod."""
        pod = PodMetrics(
            pod_id=pod_id,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            current_capital=initial_capital,
            allocated_capital=initial_capital
        )
        self.pods[pod_id] = pod
        logger.info(f"Registered pod {pod_id}: {strategy_name} with ${initial_capital:,.0f}")
        return pod

    def update_pod_pnl(
        self,
        pod_id: str,
        daily_pnl: float,
        current_equity: float
    ):
        """Update pod P&L and metrics."""
        if pod_id not in self.pods:
            return

        pod = self.pods[pod_id]

        pod.daily_pnl = daily_pnl
        pod.total_pnl += daily_pnl
        pod.mtd_pnl += daily_pnl
        pod.ytd_pnl += daily_pnl
        pod.current_capital = current_equity

        # Update drawdown
        peak = pod.initial_capital + pod.total_pnl
        if current_equity < peak:
            pod.current_drawdown = (peak - current_equity) / peak
            pod.max_drawdown = max(pod.max_drawdown, pod.current_drawdown)
        else:
            pod.current_drawdown = 0.0

        pod.last_updated = time.time()

    def check_pod_risk(self, pod_id: str) -> PodAction:
        """Check pod risk and return required action."""
        if pod_id not in self.pods:
            return PodAction(pod_id, "MAINTAIN", "Pod not found", 1.0)

        pod = self.pods[pod_id]

        # Already terminated
        if pod.status == PodStatus.TERMINATED:
            return PodAction(pod_id, "MAINTAIN", "Already terminated", 0.0)

        # Check termination threshold
        if pod.current_drawdown >= self.termination_threshold:
            pod.status = PodStatus.TERMINATED
            action = PodAction(
                pod_id=pod_id,
                action="TERMINATE",
                reason=f"Drawdown {pod.current_drawdown:.1%} >= {self.termination_threshold:.1%}",
                capital_multiplier=0.0
            )
            self.historical_actions.append(action)
            logger.warning(f"Pod {pod_id} TERMINATED: {action.reason}")
            return action

        # Check reduction threshold
        if pod.current_drawdown >= self.drawdown_cut_threshold:
            if pod.status != PodStatus.REDUCED:
                pod.status = PodStatus.REDUCED
                pod.reduction_count += 1
                pod.allocated_capital *= (1 - self.drawdown_cut_amount)

                action = PodAction(
                    pod_id=pod_id,
                    action="REDUCE",
                    reason=f"Drawdown {pod.current_drawdown:.1%} >= {self.drawdown_cut_threshold:.1%}",
                    capital_multiplier=1 - self.drawdown_cut_amount
                )
                self.historical_actions.append(action)
                logger.warning(f"Pod {pod_id} REDUCED: {action.reason}")
                return action

        # Check for recovery (reset status if drawdown cleared)
        if pod.current_drawdown < 0.02 and pod.status == PodStatus.REDUCED:
            pod.status = PodStatus.ACTIVE

        return PodAction(pod_id, "MAINTAIN", "Within risk limits", 1.0)

    def rank_pods(self) -> List[Tuple[str, float]]:
        """Rank pods by risk-adjusted performance."""
        active_pods = [
            (pod_id, pod) for pod_id, pod in self.pods.items()
            if pod.status in [PodStatus.ACTIVE, PodStatus.REDUCED]
        ]

        # Score based on Sharpe, returns, and drawdown
        def score_pod(pod: PodMetrics) -> float:
            return_score = pod.ytd_pnl / max(pod.initial_capital, 1) * 100
            sharpe_score = pod.sharpe_ratio * 30
            dd_penalty = -pod.max_drawdown * 50
            return return_score + sharpe_score + dd_penalty

        ranked = sorted(
            [(pod_id, score_pod(pod)) for pod_id, pod in active_pods],
            key=lambda x: x[1],
            reverse=True
        )

        return ranked

    def reallocate_capital(self, total_capital: float) -> Dict[str, float]:
        """
        Reallocate capital based on performance.

        Top 20% get bonus allocation
        Bottom 20% get reduced allocation
        """
        ranked = self.rank_pods()
        if not ranked:
            return {}

        n_pods = len(ranked)
        top_cutoff = max(1, int(n_pods * 0.2))
        bottom_cutoff = max(1, int(n_pods * 0.2))

        allocations = {}
        base_allocation = total_capital / n_pods

        for i, (pod_id, score) in enumerate(ranked):
            pod = self.pods[pod_id]

            if pod.status == PodStatus.TERMINATED:
                allocations[pod_id] = 0.0
            elif i < top_cutoff and pod.sharpe_ratio >= self.min_sharpe_for_bonus:
                # Top performers get bonus
                allocations[pod_id] = base_allocation * (1 + self.top_performer_bonus)
            elif i >= n_pods - bottom_cutoff:
                # Bottom performers get reduced
                allocations[pod_id] = base_allocation * 0.8
            else:
                allocations[pod_id] = base_allocation

            pod.allocated_capital = allocations[pod_id]

        return allocations

    def get_summary(self) -> Dict:
        """Get summary of all pods."""
        active = sum(1 for p in self.pods.values() if p.status == PodStatus.ACTIVE)
        reduced = sum(1 for p in self.pods.values() if p.status == PodStatus.REDUCED)
        terminated = sum(1 for p in self.pods.values() if p.status == PodStatus.TERMINATED)

        total_pnl = sum(p.total_pnl for p in self.pods.values())
        total_capital = sum(p.allocated_capital for p in self.pods.values())

        return {
            "total_pods": len(self.pods),
            "active_pods": active,
            "reduced_pods": reduced,
            "terminated_pods": terminated,
            "total_pnl": total_pnl,
            "total_capital": total_capital,
            "actions_taken": len(self.historical_actions)
        }


# Global singleton
_pod_manager: Optional[PodRiskManager] = None


def get_pod_manager() -> PodRiskManager:
    """Get or create global pod risk manager."""
    global _pod_manager
    if _pod_manager is None:
        _pod_manager = PodRiskManager()
    return _pod_manager
