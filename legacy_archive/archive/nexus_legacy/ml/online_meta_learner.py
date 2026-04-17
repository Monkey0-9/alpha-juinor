"""
Online Meta-Learning for Rapid Adaptation
=========================================

Implements online learning algorithms for model weight adaptation.

Features:
- Exponentiated Gradient (EG) for convex optimization
- Thompson Sampling for exploration
- Multi-armed bandit framework
- Contextual bandits with regime features

Phase 1.1 & 4.3: Online Agent Weight Learning
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BanditArm:
    """Represents a single arm (model/strategy)."""
    name: str
    alpha: float  # Beta prior alpha
    beta: float   # Beta prior beta
    pulls: int
    total_reward: float


class OnlineMetaLearner:
    """
    Online meta-learning for dynamic model/strategy weighting.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1
    ):
        self.lr = learning_rate
        self.eps = exploration_rate
        self.arms: Dict[str, BanditArm] = {}
        self.weights: Dict[str, float] = {}
        self.history: List[Dict] = []
        logger.info(
            f"Meta-Learner init: lr={learning_rate}, eps={exploration_rate}"
        )

    def register_arm(self, name: str):
        """Register a new arm (model/strategy)."""
        self.arms[name] = BanditArm(
            name=name,
            alpha=1.0,
            beta=1.0,
            pulls=0,
            total_reward=0.0
        )
        self.weights[name] = 1.0 / max(1, len(self.arms))
        self._normalize_weights()

    def _normalize_weights(self):
        """Ensure weights sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    def thompson_sample(self) -> str:
        """
        Thompson Sampling: Select arm by sampling from posteriors.
        """
        if not self.arms:
            raise ValueError("No arms registered")

        samples = {}
        for name, arm in self.arms.items():
            samples[name] = np.random.beta(arm.alpha, arm.beta)

        return max(samples, key=samples.get)

    def update_arm(self, name: str, reward: float):
        """
        Update arm statistics after observing reward.

        Args:
            name: Arm name
            reward: Reward in [0, 1]
        """
        if name not in self.arms:
            logger.warning(f"Unknown arm: {name}")
            return

        arm = self.arms[name]
        arm.pulls += 1
        arm.total_reward += reward

        # Update Beta distribution parameters
        if reward > 0.5:
            arm.alpha += reward
        else:
            arm.beta += (1 - reward)

        self.history.append({
            "arm": name,
            "reward": reward,
            "pulls": arm.pulls
        })

    def exponentiated_gradient_update(
        self,
        losses: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Exponentiated Gradient (EG) weight update.

        w_t+1(i) = w_t(i) * exp(-eta * loss_t(i)) / Z

        Args:
            losses: Dict of arm name -> loss value

        Returns:
            Updated weights
        """
        for name, loss in losses.items():
            if name in self.weights:
                self.weights[name] *= np.exp(-self.lr * loss)

        self._normalize_weights()
        return self.weights.copy()

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()

    def select_action(self, context: Optional[np.ndarray] = None) -> str:
        """
        Select action using epsilon-greedy with Thompson fallback.
        """
        if np.random.random() < self.eps:
            # Explore: random selection
            return np.random.choice(list(self.arms.keys()))
        else:
            # Exploit: Thompson Sampling
            return self.thompson_sample()

    def get_ucb_scores(self, c: float = 2.0) -> Dict[str, float]:
        """
        Get Upper Confidence Bound scores for each arm.
        """
        total_pulls = sum(a.pulls for a in self.arms.values())
        scores = {}

        for name, arm in self.arms.items():
            if arm.pulls == 0:
                scores[name] = float('inf')
            else:
                mean_reward = arm.total_reward / arm.pulls
                exploration_bonus = c * np.sqrt(
                    np.log(total_pulls + 1) / arm.pulls
                )
                scores[name] = mean_reward + exploration_bonus

        return scores


# Singleton
_meta_learner = None


def get_meta_learner() -> OnlineMetaLearner:
    global _meta_learner
    if _meta_learner is None:
        _meta_learner = OnlineMetaLearner()
    return _meta_learner
