import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """Rule-based order execution router.

    Selects execution tactics (wait, small order, large order) based
    on observable market microstructure features rather than a trained
    neural network.  This is deliberately simple and transparent —
    no randomness is involved in production decisions.

    Actions:
        0 — Wait  (defer execution to next cycle)
        1 — Small Market Order  (conservative fill)
        2 — Large Market Order  (aggressive fill)
    """

    def __init__(self) -> None:
        self.cumulative_reward: float = 0.0
        self.decision_count: int = 0

    def get_action(self, market_state: np.ndarray[Any, Any]) -> int:
        """Determine execution tactic from market microstructure.

        Parameters
        ----------
        market_state : np.ndarray
            Feature vector. When a full feature pipeline is wired,
            indices should map to:
                [0] spread_bps, [1] volume_ratio, [2] volatility,
                [3] momentum.
            If the vector is too short or all zeros, default to
            conservative execution (action 1).
        """
        if market_state is None or len(market_state) < 4:
            return 1  # Conservative default

        spread_bps = float(market_state[0])
        volume_ratio = float(market_state[1])
        volatility = float(market_state[2])
        momentum = float(market_state[3])

        if volatility > 0.035 or (spread_bps > 0.001 and volume_ratio < 0.7):
            return 0

        if abs(momentum) > 0.025:
            return 2

        return 1

    def learn(self, reward: float) -> None:
        """Record trade outcome for monitoring.

        This does NOT perform gradient updates — it tracks
        cumulative reward so the operator can evaluate execution
        quality over time.
        """
        self.cumulative_reward += reward
        self.decision_count += 1
        avg = (
            self.cumulative_reward / self.decision_count
            if self.decision_count
            else 0.0
        )
        logger.debug(
            f"Execution feedback — Reward: {reward:.4f}, "
            f"Cumulative avg: {avg:.4f}, "
            f"Decisions: {self.decision_count}"
        )
