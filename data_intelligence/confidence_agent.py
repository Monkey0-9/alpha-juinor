
import json
import logging
import os
import threading
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfidenceAgent:
    """
    Tracks and updates Per-Provider and Per-Symbol confidence scores.
    Uses simple Bayesian-like decay:
    C_{t+1} = alpha * C_t + (1 - alpha) * success_score
    """
    def __init__(self, state_file="runtime/provider_confidence.json", alpha=0.9):
        self.state_file = state_file
        self.alpha = alpha
        self.lock = threading.Lock()
        self.confidence_scores: Dict[str, float] = {}
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.confidence_scores = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load confidence state: {e}")
                self.confidence_scores = {}

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.confidence_scores, f)
        except Exception as e:
            logger.error(f"Failed to persist confidence state: {e}")

    def get_provider_confidence(self, provider_name: str) -> float:
        with self.lock:
            return self.confidence_scores.get(provider_name, 0.5)

    def update_confidence(self, provider_name: str, success: bool, latency_ms: float = 0.0):
        """
        Update confidence based on fetch outcome.
        """
        score = 1.0 if success else 0.0

        # Penalize high latency? (Optional)
        # if latency_ms > 2000: score *= 0.8

        with self.lock:
            current = self.confidence_scores.get(provider_name, 0.5)
            # EMA Update
            new_score = self.alpha * current + (1.0 - self.alpha) * score
            self.confidence_scores[provider_name] = new_score

        # Periodic save? For now, save on every update (slow but safe) or manage separately.
        # Ideally, save async.
        # self._save_state()

    def persist(self):
        with self.lock:
            self._save_state()
