import numpy as np
import structlog
from typing import List, Dict

logger = structlog.get_logger()

class ProviderBandit:
    """
    Epsilon-Greedy Multi-Armed Bandit for data provider selection.
    Picks provider based on historical success rate and latency.
    """
    def __init__(self, providers: List[str], epsilon: float = 0.1):
        self.providers = providers
        self.epsilon = epsilon
        self.counts = {p: 0 for p in providers}
        self.values = {p: 0.0 for p in providers} # Success rate

    def select_provider(self) -> str:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.providers)

        # Greedy choice
        return max(self.values, key=self.values.get)

    def update(self, provider: str, success: bool):
        self.counts[provider] += 1
        n = self.counts[provider]
        value = self.values[provider]
        reward = 1.0 if success else 0.0

        # New value = old value + (1/n) * (reward - old value)
        self.values[provider] = value + (1/n) * (reward - value)

        logger.info("Bandit updated", provider=provider, success=success, new_rate=self.values[provider])
