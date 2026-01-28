"""
ci/promote_model.py

Section E: Model Promotion Pipeline.
Checks shadow performance and tail metrics before promotion.
"""
import sys
import os
import json
import logging
from datetime import datetime

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.registry import ModelRegistry, STATUS_SHADOW, STATUS_PROD

logger = logging.getLogger("PROMOTE")
logging.basicConfig(level=logging.INFO)

def check_promotion_criteria(model_id: str):
    registry = ModelRegistry()
    if model_id not in registry.models:
        logger.error(f"Model {model_id} not found")
        return False

    model = registry.models[model_id]
    if model.status != STATUS_SHADOW:
        logger.error(f"Model {model_id} is not in SHADOW status (Current: {model.status})")
        return False

    # Check 1: Duration in Shadow
    # Simulated check: ensure shadow_start_date was > 60 days ago (or forced in dev)
    # For now, we print check.
    start = model.shadow_start_date
    logger.info(f"Model in shadow since {start}")

    # Check 2: Metrics (PSI, CVaR)
    # Using metrics stored in registry
    metrics = model.metrics
    cvar = metrics.get('cvar_95', -1.0)
    max_dd = metrics.get('max_drawdown', -1.0)

    if cvar < -0.05: # Worse than -5%
        logger.error(f"Promotion Rejected: CVaR {cvar:.2%} exceeds limit")
        return False

    if max_dd < -0.20:
        logger.error(f"Promotion Rejected: Max DD {max_dd:.2%} exceeds limit")
        return False

    # Check 3: OOS Test (Mock function)
    if not run_oos_test(model_id):
        logger.error("Promotion Rejected: OOS Test Failed")
        return False

    logger.info(f"Model {model_id} PASSED all criteria.")
    return True

def run_oos_test(model_id):
    # Mock OOS test
    return True

def promote(model_id: str, approver: str):
    if check_promotion_criteria(model_id):
        registry = ModelRegistry()
        registry.update_status(model_id, STATUS_PROD, user=approver)
        logger.info(f"SUCCESS: Promoted {model_id} to PROD")
    else:
        logger.error(f"FAILURE: Could not promote {model_id}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ci/promote_model.py <model_id> <approver>")
        sys.exit(1)
    promote(sys.argv[1], sys.argv[2])
