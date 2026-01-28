"""
ml/governor.py

Section E: Model Governor.
Drift Detection (PSI), Calibration, Disagreement.
"""
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger("ML_GOV")

class ModelGovernor:
    def __init__(self, psi_threshold=0.2):
        self.psi_threshold = psi_threshold

    def calculate_psi(self, expected: np.array, actual: np.array, buckets=10) -> float:
        """
        Calculate Population Stability Index (PSI).
        """
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        # Determine buckets based on expected
        expected_percents = np.percentile(expected, breakpoints)

        expected_cnts, _ = np.histogram(expected, expected_percents)
        actual_cnts, _ = np.histogram(actual, expected_percents)

        # Avoid zero division
        expected_cnts = np.where(expected_cnts == 0, 1e-6, expected_cnts)
        actual_cnts = np.where(actual_cnts == 0, 1e-6, actual_cnts)

        expected_dist = expected_cnts / len(expected)
        actual_dist = actual_cnts / len(actual)

        psi_values = (expected_dist - actual_dist) * np.log(expected_dist / actual_dist)
        psi = np.sum(psi_values)

        return float(psi)

    def check_drift(self, model_id: str, baseline_features: np.array, current_features: np.array) -> Dict:
        """
        Check for feature drift.
        Returns: {drift_detected: bool, psi: float}
        """
        psi = self.calculate_psi(baseline_features, current_features)

        drift = psi > self.psi_threshold
        if drift:
            logger.warning(f"[MODEL_DRIFT] Model {model_id} PSI={psi:.4f} > {self.psi_threshold}")

        return {"drift_detected": drift, "psi": psi}

    def check_disagreement(self, predictions: List[float]) -> float:
        """
        Variance of predictions from ensemble.
        """
        return float(np.var(predictions))
