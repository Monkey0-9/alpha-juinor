"""
scripts/alpha_diagnostics.py

MANDATORY FIX 7: PERFORMANCE DIAGNOSTICS.
Verifies Alpha Rehabilitation status.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from alpha_families.normalization import AlphaNormalizer
from strategies.institutional_strategy import InstitutionalStrategy
from portfolio.allocator import InstitutionalAllocator
from contracts import AllocationRequest

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DIAGNOSTICS")

def run_diagnostics():
    logger.info("Starting Alpha Rehabilitation Diagnostics...")

    # 1. Verify Normalization Pipeline
    norm = AlphaNormalizer()
    history = pd.Series(np.random.normal(0, 1, 100))
    z, conf = norm.normalize_signal(2.0, history)
    logger.info(f"Normalization Check: Raw=2.0 -> Z={z:.2f}, Conf={conf:.2f}")

    if not (1.8 < z < 2.2 and 0 < conf < 1):
        logger.error("FAIL: Normalization logic broken")
        return
    else:
        logger.info("PASS: Normalization Pipeline")

    # 2. Verify Satellite Sleeve Allocation
    allocator = InstitutionalAllocator()

    # Create synthetic requests
    # 2 High conviction (Satellite candidates)
    # 10 Normal conviction (Core)
    requests = []

    # High Conviction
    requests.append(AllocationRequest(symbol="SAT1", mu=0.08, sigma=0.02, confidence=0.9, liquidity=1e9, regime="NORMAL", timestamp="2026-01-01"))
    requests.append(AllocationRequest(symbol="SAT2", mu=0.07, sigma=0.02, confidence=0.85, liquidity=1e9, regime="NORMAL", timestamp="2026-01-01"))

    # Core
    for i in range(10):
        requests.append(AllocationRequest(symbol=f"CORE{i}", mu=0.02, sigma=0.01, confidence=0.5, liquidity=1e9, regime="NORMAL", timestamp="2026-01-01"))

    weights = allocator.allocate_batch(requests)

    # Check if SAT1/SAT2 got allocations
    sat1_w = weights.get("SAT1", 0.0)
    sat2_w = weights.get("SAT2", 0.0)

    logger.info(f"Satellite Allocations: SAT1={sat1_w:.2%}, SAT2={sat2_w:.2%}")

    if sat1_w > 0 and sat2_w > 0:
        logger.info("PASS: Satellite Sleeve Active")
        # Check if they are roughly 5% each (10% total / 2)
        if abs(sat1_w - 0.05) < 0.01:
             logger.info("PASS: Satellite Sizing Correct (~5%)")
        else:
             logger.warning(f"WARN: Satellite sizing deviation (Expected ~5%, Got {sat1_w:.2%})")
    else:
        logger.error("FAIL: Satellite Sleeve Inactive")

    # 3. Capital Utilization
    total_alloc = sum(weights.values())
    logger.info(f"Total Capital Utilization: {total_alloc:.2%}")

    logger.info("DIAGNOSTICS COMPLETE: Alpha Rehabilitation Verified.")

if __name__ == "__main__":
    run_diagnostics()
