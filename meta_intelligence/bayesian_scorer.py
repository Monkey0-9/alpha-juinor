
import os
import json
import logging
import threading
from typing import Dict

logger = logging.getLogger(__name__)

class BayesianScorer:
    """
    Tracks and updates agent weights using Bayesian updating (Beta distribution).
    Weights are derived as the expected value of the posterior distribution.
    """
    def __init__(self, storage_path: str = "data/agent_performance.json"):
        self.storage_path = storage_path
        self.lock = threading.Lock()
        self.performance = self._load()

        # Bayesian Prior: Alpha=2, Beta=2 (Start with 0.5 mean, but some weight/inertia)
        self.prior_alpha = 2
        self.prior_beta = 2

    def _load(self) -> Dict[str, Dict[str, int]]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load performance data: {e}")
        return {}

    def save(self):
        with self.lock:
            try:
                os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
                with open(self.storage_path, 'w') as f:
                    json.dump(self.performance, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save performance data: {e}")

    def get_weight(self, agent_name: str) -> float:
        """Calculate weight using Bayesian mean: (hits + alpha) / (total + alpha + beta)"""
        stats = self.performance.get(agent_name, {"hits": 0, "total": 0})
        hits = stats["hits"]
        total = stats["total"]

        weight = (hits + self.prior_alpha) / (total + self.prior_alpha + self.prior_beta)
        return float(weight)

    def update_performance(self, agent_results: Dict[str, float], actual_return: float):
        """
        Update performance metrics based on realized return.
        agent_results: Dict of {agent_name: mu}
        """
        with self.lock:
            for name, mu in agent_results.items():
                if name not in self.performance:
                    self.performance[name] = {"hits": 0, "total": 0}

                # A "hit" is defined as predicting the sign correctly
                # (Simple approach for price directionality)
                if (mu > 0 and actual_return > 0) or (mu < 0 and actual_return < 0):
                    self.performance[name]["hits"] += 1

                self.performance[name]["total"] += 1

        self.save()

    def get_report(self) -> Dict[str, float]:
        """Returns a sorted dictionary of weights for all known agents."""
        weights = {name: self.get_weight(name) for name in self.performance}
        return dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))
