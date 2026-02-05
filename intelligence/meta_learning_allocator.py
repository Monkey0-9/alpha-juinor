"""
Meta-Learning Allocator - Contextual Bandit Strategy Selection
================================================================

A self-learning allocator that dynamically weights strategies
based on their performance in the current market context.

Each strategy = an "arm" in contextual bandit
Context = current market regime
Reward = strategy returns

The allocator LEARNS which strategy works best in which context.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from collections import defaultdict
import numpy as np
import threading

logger = logging.getLogger(__name__)


class AllocationMode(Enum):
    """Allocation mode."""
    EXPLOIT = "EXPLOIT"  # Use learned weights
    EXPLORE = "EXPLORE"  # Try underweighted strategies
    BALANCED = "BALANCED"  # Balance exploration/exploitation


@dataclass
class StrategyArm:
    """A strategy treated as a bandit arm."""
    strategy_name: str

    # Reward statistics per regime
    regime_rewards: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    regime_pulls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Overall statistics
    total_pulls: int = 0
    total_reward: float = 0.0

    # Thompson Sampling parameters (per regime)
    alpha: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))  # Successes + 1
    beta: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))   # Failures + 1

    # Current weight
    current_weight: float = 0.0


@dataclass
class AllocationDecision:
    """Allocation decision for capital distribution."""
    timestamp: datetime
    regime: str
    mode: AllocationMode

    # Allocations
    allocations: Dict[str, float]  # Strategy -> Weight (sum = 1.0)

    # Context
    exploration_rate: float

    # Reasoning
    reasoning: List[str]


class ThompsonSamplingBandit:
    """
    Thompson Sampling for contextual bandit.

    Balances exploration/exploitation automatically.
    """

    def __init__(self, decay_rate: float = 0.999):
        """Initialize bandit."""
        self.arms: Dict[str, StrategyArm] = {}
        self.decay_rate = decay_rate  # Reward decay for recency
        self._lock = threading.Lock()

    def add_arm(self, strategy_name: str):
        """Add a strategy arm."""
        with self._lock:
            if strategy_name not in self.arms:
                self.arms[strategy_name] = StrategyArm(strategy_name=strategy_name)

    def pull_arm(self, strategy_name: str, regime: str) -> float:
        """
        Pull an arm and get sampled expected reward.

        Returns a sample from the Beta distribution for this arm.
        """
        with self._lock:
            if strategy_name not in self.arms:
                return 0.5  # Uniform prior

            arm = self.arms[strategy_name]

            # Sample from Beta distribution
            alpha = arm.alpha.get(regime, 1.0)
            beta_param = arm.beta.get(regime, 1.0)

            return np.random.beta(alpha, beta_param)

    def update_arm(
        self,
        strategy_name: str,
        regime: str,
        reward: float,
        success_threshold: float = 0.0
    ):
        """
        Update arm with observed reward.

        Args:
            strategy_name: Strategy that was used
            regime: Market regime at execution
            reward: Actual return (can be negative)
            success_threshold: Threshold for binary success
        """
        with self._lock:
            if strategy_name not in self.arms:
                self.add_arm(strategy_name)

            arm = self.arms[strategy_name]

            # Record raw reward
            arm.regime_rewards[regime].append(reward)
            arm.regime_pulls[regime] += 1
            arm.total_pulls += 1
            arm.total_reward += reward

            # Update Beta parameters
            # Binary: success if reward > threshold
            if reward > success_threshold:
                arm.alpha[regime] = arm.alpha.get(regime, 1.0) + 1
            else:
                arm.beta[regime] = arm.beta.get(regime, 1.0) + 1

            # Apply decay to keep recent data relevant
            arm.alpha[regime] *= self.decay_rate
            arm.beta[regime] *= self.decay_rate

            # Ensure minimum values
            arm.alpha[regime] = max(1.0, arm.alpha[regime])
            arm.beta[regime] = max(1.0, arm.beta[regime])

    def get_expected_rewards(self, regime: str) -> Dict[str, float]:
        """Get expected reward for each arm in given regime."""
        with self._lock:
            expectations = {}

            for name, arm in self.arms.items():
                alpha = arm.alpha.get(regime, 1.0)
                beta_param = arm.beta.get(regime, 1.0)

                # Mean of Beta distribution
                expectations[name] = alpha / (alpha + beta_param)

            return expectations


class MetaLearningAllocator:
    """
    Meta-learning allocator using contextual bandits.

    Dynamically allocates capital to strategies based on:
    1. Current market regime (context)
    2. Historical performance in that regime (reward)
    3. Exploration/exploitation balance
    """

    def __init__(
        self,
        exploration_rate: float = 0.1,
        min_allocation: float = 0.05,
        max_allocation: float = 0.40
    ):
        """Initialize the allocator."""
        self.bandit = ThompsonSamplingBandit()
        self.exploration_rate = exploration_rate
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation

        # Regime history
        self.regime_history: List[Tuple[datetime, str]] = []

        # Allocation history
        self.allocation_history: List[AllocationDecision] = []

        self._lock = threading.Lock()

        logger.info(
            f"[META-ALLOC] Meta-Learning Allocator initialized | "
            f"Exploration: {exploration_rate:.0%}"
        )

    def register_strategy(self, strategy_name: str):
        """Register a strategy for allocation."""
        self.bandit.add_arm(strategy_name)
        logger.info(f"[META-ALLOC] Strategy registered: {strategy_name}")

    def record_result(
        self,
        strategy_name: str,
        regime: str,
        return_pct: float
    ):
        """
        Record the result of using a strategy in a regime.

        This is how the allocator LEARNS.
        """
        self.bandit.update_arm(strategy_name, regime, return_pct)

        logger.debug(
            f"[META-ALLOC] Learning: {strategy_name} in {regime} "
            f"-> {return_pct:.2%}"
        )

    def allocate(
        self,
        regime: str,
        available_strategies: List[str],
        mode: AllocationMode = AllocationMode.BALANCED
    ) -> AllocationDecision:
        """
        Allocate capital across strategies for current regime.

        Returns weights that sum to 1.0.
        """
        if not available_strategies:
            return self._empty_allocation(regime)

        # Record regime
        self.regime_history.append((datetime.utcnow(), regime))

        # Get sampled values from Thompson Sampling
        samples = {}
        for strategy in available_strategies:
            samples[strategy] = self.bandit.pull_arm(strategy, regime)

        # Exploration: sometimes pick underexplored arms
        if mode == AllocationMode.EXPLORE or (
            mode == AllocationMode.BALANCED and
            np.random.random() < self.exploration_rate
        ):
            allocations = self._exploration_allocation(
                available_strategies, regime, samples
            )
            reasoning = ["Exploration mode: trying underweighted strategies"]
        else:
            allocations = self._exploitation_allocation(
                samples, available_strategies
            )
            reasoning = ["Exploitation mode: using learned weights"]

        # Apply constraints
        allocations = self._apply_constraints(allocations)

        # Add reasoning
        expected = self.bandit.get_expected_rewards(regime)
        top_strategy = max(expected.items(), key=lambda x: x[1])[0] if expected else "N/A"
        reasoning.append(f"Best strategy for {regime}: {top_strategy}")

        decision = AllocationDecision(
            timestamp=datetime.utcnow(),
            regime=regime,
            mode=mode,
            allocations=allocations,
            exploration_rate=self.exploration_rate,
            reasoning=reasoning
        )

        with self._lock:
            self.allocation_history.append(decision)

        logger.info(
            f"[META-ALLOC] Allocation for {regime}: "
            f"{', '.join(f'{s}:{w:.0%}' for s, w in allocations.items())}"
        )

        return decision

    def _exploration_allocation(
        self,
        strategies: List[str],
        regime: str,
        samples: Dict[str, float]
    ) -> Dict[str, float]:
        """Allocate with exploration bias."""
        # Find underexplored strategies
        with self._lock:
            pulls = {
                s: self.bandit.arms.get(s, StrategyArm(s)).regime_pulls.get(regime, 0)
                for s in strategies
            }

        min_pulls = min(pulls.values()) if pulls else 0

        # Boost underexplored
        weights = {}
        for s in strategies:
            base_weight = samples.get(s, 0.5)
            exploration_boost = 1.0 / (1 + pulls.get(s, 0) - min_pulls)
            weights[s] = base_weight * (1 + exploration_boost)

        # Normalize
        total = sum(weights.values())
        return {s: w / total for s, w in weights.items()}

    def _exploitation_allocation(
        self,
        samples: Dict[str, float],
        strategies: List[str]
    ) -> Dict[str, float]:
        """Allocate based on Thompson samples."""
        # Softmax over samples for smooth allocation
        temps = np.array([samples.get(s, 0.5) for s in strategies])

        # Temperature scaling
        temperature = 0.5  # Higher = more uniform
        exp_temps = np.exp(temps / temperature)
        softmax = exp_temps / exp_temps.sum()

        return {s: w for s, w in zip(strategies, softmax)}

    def _apply_constraints(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max allocation constraints."""
        constrained = {}

        for s, w in allocations.items():
            constrained[s] = max(self.min_allocation, min(self.max_allocation, w))

        # Renormalize
        total = sum(constrained.values())
        return {s: w / total for s, w in constrained.items()}

    def _empty_allocation(self, regime: str) -> AllocationDecision:
        """Return empty allocation."""
        return AllocationDecision(
            timestamp=datetime.utcnow(),
            regime=regime,
            mode=AllocationMode.BALANCED,
            allocations={},
            exploration_rate=self.exploration_rate,
            reasoning=["No strategies available"]
        )

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of what the allocator has learned."""
        with self._lock:
            return {
                "strategies_tracked": len(self.bandit.arms),
                "total_allocations": len(self.allocation_history),
                "arms": {
                    name: {
                        "total_pulls": arm.total_pulls,
                        "total_reward": arm.total_reward,
                        "avg_reward": arm.total_reward / arm.total_pulls if arm.total_pulls > 0 else 0,
                        "regime_pulls": dict(arm.regime_pulls)
                    }
                    for name, arm in self.bandit.arms.items()
                }
            }

    def get_regime_preferences(self, regime: str) -> Dict[str, float]:
        """Get learned strategy preferences for a regime."""
        return self.bandit.get_expected_rewards(regime)


class UCBAllocator:
    """
    Upper Confidence Bound allocator.

    Alternative to Thompson Sampling - more deterministic.
    """

    def __init__(self, c: float = 2.0):
        """Initialize UCB allocator."""
        self.c = c  # Exploration parameter
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.values: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.total_pulls = 0

    def get_ucb_value(self, strategy: str, regime: str) -> float:
        """Calculate UCB value for a strategy."""
        n = self.counts[regime][strategy]

        if n == 0:
            return float('inf')  # Explore first

        avg_value = self.values[regime][strategy] / n
        exploration_bonus = self.c * np.sqrt(np.log(self.total_pulls + 1) / n)

        return avg_value + exploration_bonus

    def select(self, strategies: List[str], regime: str) -> str:
        """Select best strategy using UCB."""
        ucb_values = {s: self.get_ucb_value(s, regime) for s in strategies}
        return max(ucb_values.items(), key=lambda x: x[1])[0]

    def update(self, strategy: str, regime: str, reward: float):
        """Update with observed reward."""
        self.counts[regime][strategy] += 1
        self.values[regime][strategy] += reward
        self.total_pulls += 1


# Singleton
_allocator: Optional[MetaLearningAllocator] = None


def get_meta_allocator() -> MetaLearningAllocator:
    """Get or create the Meta-Learning Allocator."""
    global _allocator
    if _allocator is None:
        _allocator = MetaLearningAllocator()
    return _allocator
