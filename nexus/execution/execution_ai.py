import logging
import numpy as np
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    action: int
    order_type: str
    size_pct: float
    urgency: str
    limit_price: Optional[float] = None
    twap_duration: Optional[int] = None


class VWAPExecutionModel:
    def __init__(self, target_participation: float = 0.1, min_bins: int = 13):
        self.target_participation = target_participation
        self.min_bins = min_bins

    def schedule(self, total_qty: int, volume_profile: List[float]) -> List[int]:
        profile = (
            np.array(volume_profile[-self.min_bins :])
            if len(volume_profile) >= self.min_bins
            else np.ones(self.min_bins)
        )
        profile = profile / profile.sum()
        return [int(total_qty * w) for w in profile]


class TWAPExecutionModel:
    def schedule(self, total_qty: int, num_slices: int = 10) -> List[int]:
        base = total_qty // num_slices
        remainder = total_qty % num_slices
        return [base + (1 if i < remainder else 0) for i in range(num_slices)]


class ImplementationShortfallModel:
    def __init__(self, urgency: float = 0.5, alpha_decay: float = 0.1):
        self.urgency = urgency
        self.alpha_decay = alpha_decay

    def optimal_participation(
        self, alpha: float, volatility: float, spread: float
    ) -> float:
        if volatility < 1e-9:
            return 0.1
        expected_cost = spread / 2
        alpha_urgency = abs(alpha) * self.urgency
        participation = alpha_urgency / (
            expected_cost + volatility * self.alpha_decay + 1e-9
        )
        return float(np.clip(participation, 0.01, 0.5))


class ExecutionAgent:
    def __init__(self, default_slippage_bps: float = 2.0):
        self.cumulative_reward: float = 0.0
        self.decision_count: int = 0
        self.default_slippage = default_slippage_bps
        self.vwap_model = VWAPExecutionModel()
        self.twap_model = TWAPExecutionModel()
        self.shortfall_model = ImplementationShortfallModel()

    def select_execution_model(
        self,
        market_state: np.ndarray,
        alpha: float = 0.0,
        volume_profile: Optional[List[float]] = None,
    ) -> ExecutionPlan:
        if market_state is None or len(market_state) < 5:
            return ExecutionPlan(
                action=1, order_type="MARKET", size_pct=0.5, urgency="NORMAL"
            )
        volatility = float(market_state[2]) if len(market_state) > 2 else 0.01
        momentum = float(market_state[3]) if len(market_state) > 3 else 0.0
        spread_bps = (
            float(market_state[0]) * 10000
            if len(market_state) > 0
            else self.default_slippage
        )
        volume_ratio = float(market_state[1]) if len(market_state) > 1 else 1.0
        participation = self.shortfall_model.optimal_participation(
            alpha, volatility, spread_bps / 10000
        )

        if volatility > 0.04 or spread_bps > 10:
            return ExecutionPlan(
                action=0, order_type="WAIT", size_pct=0.0, urgency="LOW"
            )
        if abs(momentum) > 0.03 and alpha * momentum > 0:
            return ExecutionPlan(
                action=2,
                order_type="MARKET",
                size_pct=min(1.0, participation * 2),
                urgency="HIGH",
            )
        if spread_bps < 3 and volume_ratio > 1.5:
            if volume_profile and len(volume_profile) >= 13:
                return ExecutionPlan(
                    action=1,
                    order_type="VWAP",
                    size_pct=participation,
                    urgency="MEDIUM",
                    twap_duration=13,
                )
            return ExecutionPlan(
                action=1,
                order_type="MARKET",
                size_pct=participation,
                urgency="MEDIUM",
            )
        if spread_bps < 5:
            return ExecutionPlan(
                action=1,
                order_type="TWAP",
                size_pct=participation * 0.7,
                urgency="NORMAL",
                twap_duration=10,
            )
        return ExecutionPlan(
            action=0,
            order_type="LIMIT",
            size_pct=participation * 0.3,
            urgency="LOW",
        )

    def get_action(
        self,
        market_state: np.ndarray,
        alpha: float = 0.0,
        volume_profile: Optional[List[float]] = None,
    ) -> int:
        plan = self.select_execution_model(market_state, alpha, volume_profile)
        return plan.action

    def learn(self, reward: float) -> None:
        self.cumulative_reward += reward
        self.decision_count += 1
        avg = (
            self.cumulative_reward / self.decision_count if self.decision_count else 0.0
        )
        logger.debug(
            "Execution feedback — Reward: %.4f, Cumulative avg: %.4f, Decisions: %d",
            reward,
            avg,
            self.decision_count,
        )
