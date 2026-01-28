
import sys
import os
import logging
import numpy as np
import pandas as pd

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.drift_detection import DriftDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DriftValidation")

def run_validation():
    logger.info("Starting Drift Validation Job...")

    # 1. Simulate Baseline Data (e.g. Training Set)
    # Normal distribution: mean=0, std=1
    baseline = np.random.normal(0, 1, 1000)

    # 2. Simulate Production Data (No Drift)
    prod_ok = np.random.normal(0, 1, 1000)

    # 3. Simulate Production Data (Drifted)
    # Shifted mean=0.5
    prod_drift = np.random.normal(0.5, 1, 1000)

    detector = DriftDetector()

    # Check OK
    psi_ok = detector.calculate_psi(baseline, prod_ok)
    logger.info(f"Test 1 (No Drift): PSI = {psi_ok:.4f} (Threshold 0.1)")
    if psi_ok < 0.1:
        logger.info("PASS: No significant drift detected.")
    else:
        logger.warning("FAIL: Unexpected drift.")

    # Check Drift
    psi_drift = detector.calculate_psi(baseline, prod_drift)
    logger.info(f"Test 2 (Simulated Drift): PSI = {psi_drift:.4f}")
    if psi_drift > 0.1:
        logger.info("PASS: Drift correctly detected.")
    else:
        logger.warning("FAIL: Drift NOT detected.")

    # Report
    if psi_ok < 0.1 and psi_drift > 0.1:
        logger.info("=== VALIDATION SUCCESSFUL ===")
    else:
        logger.error("=== VALIDATION FAILED ===")

if __name__ == "__main__":
    run_validation()
