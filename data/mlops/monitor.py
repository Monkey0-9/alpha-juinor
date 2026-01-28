import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("MLOPS_MONITOR")

class DriftMonitor:
    """
    Institutional MLOps Drift Monitor.
    Calculates Population Stability Index (PSI) and monitoring metrics.
    """
    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate the PSI (Population Stability Index) for two distributions.
        """
        def scale_range(input_array, min_val, max_val):
            return (input_array - min_val) / (max_val - min_val)

        # Handle empty/invalid
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create buckets based on expected
        breakpoints = np.linspace(np.min(expected), np.max(expected), buckets + 1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Avoid division by zero
        expected_percents = np.clip(expected_percents, 0.0001, 1.0)
        actual_percents = np.clip(actual_percents, 0.0001, 1.0)

        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi_value)

    def monitor_features(self,
                         symbol: str,
                         current_features: Dict[str, float],
                         reference_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for drift in current features compared to reference distribution.
        """
        drift_report = {
            "symbol": symbol,
            "drift_detected": False,
            "metrics": {},
            "status": "GREEN"
        }

        total_psi = 0.0
        monitored_cols = 0

        for feat_name, current_val in current_features.items():
            if feat_name in reference_features.columns:
                expected = reference_features[feat_name].values
                actual = np.array([current_val] * 10) # Mock small distribution for single point check or use window

                # In production, we usually compare a window (e.g. last 100 observations)
                # For per-cycle check, we use Z-score or simple range check if only one point.
                # Let's perform a Z-score check for single point.

                mean = np.mean(expected)
                std = np.std(expected) + 1e-6
                z_score = abs(current_val - mean) / std

                drift_report["metrics"][feat_name] = {
                    "z_score": float(z_score),
                    "drift": z_score > 3.0 # Simple 3-sigma drift
                }

                if z_score > 3.0:
                    drift_report["drift_detected"] = True

        if drift_report["drift_detected"]:
            drift_report["status"] = "AMBER"
            logger.warning(f"MLOPS_DRIFT_DETECTED: {symbol} shows feature drift.")

        return drift_report
