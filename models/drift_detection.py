
import numpy as np
import logging
from typing import Dict, List, Union

logger = logging.getLogger("DriftDetector")

class DriftDetector:
    """
    Detects data drift using Population Stability Index (PSI).
    """

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI Rules of Thumb:
        - < 0.1: No significant drift
        - 0.1 - 0.2: Moderate drift
        - > 0.2: Significant drift

        Args:
            expected: Baseline/Training distribution
            actual: Current/Production distribution
            buckets: Number of bins for histogram

        Returns:
            PSI score
        """
        def scale_range(input, min_val, max_val):
            input = ((input - min_val) / (max_val - min_val))
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = np.percentile(expected, breakpoints)

        # Guard against zero range
        if breakpoints[0] == breakpoints[-1]:
             return 0.0 # No variance, assuming no drift if matching constant but technically undefined

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        # Avoid zero division
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value

    @staticmethod
    def check_drift(features_ref: Dict[str, np.ndarray], features_curr: Dict[str, np.ndarray], threshold: float = 0.2) -> Dict[str, float]:
        """
        Check drift for multiple features.

        Returns:
            Dict of {feature: psi_score} for features exceeding threshold (or all?)
        """
        drift_report = {}
        for feature_name, ref_data in features_ref.items():
            if feature_name in features_curr:
                curr_data = features_curr[feature_name]
                try:
                    psi = DriftDetector.calculate_psi(ref_data, curr_data)
                    drift_report[feature_name] = psi
                    if psi > threshold:
                        logger.warning(f"[DRIFT] Feature {feature_name} DRIFT DETECTED! PSI={psi:.4f}")
                except Exception as e:
                    logger.error(f"[DRIFT] Error checking {feature_name}: {e}")

        return drift_report
