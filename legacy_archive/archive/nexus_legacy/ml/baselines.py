"""
ml/baselines.py

Baseline model implementations for comparison and validation.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MeanBaselineModel:
    """Simple mean-based baseline."""
    def __init__(self):
        self.mean = 0.0
        logger.info("[MeanBaseline] Initialized")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit baseline to data."""
        self.mean = np.mean(y)
        logger.info(f"[MeanBaseline] Fitted with mean={self.mean:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return mean prediction for all samples."""
        return np.full(len(X), self.mean)


class BuyHoldBaseline:
    """Buy and hold baseline strategy."""
    def __init__(self):
        logger.info("[BuyHoldBaseline] Initialized")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Always predict 'hold' (return 0)."""
        return np.zeros(len(X))


class RandomBaseline:
    """Random trading baseline."""
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        logger.info("[RandomBaseline] Initialized")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return random predictions."""
        return np.random.randn(len(X))


def get_baselines() -> Dict[str, Any]:
    """
    Get all baseline models for comparison.

    Returns:
        Dictionary of baseline model instances
    """
    baselines = {
        "mean": MeanBaselineModel(),
        "buy_hold": BuyHoldBaseline(),
        "random": RandomBaseline(),
    }
    logger.info(f"[BaselinesFactory] Created {len(baselines)} baselines")
    return baselines
