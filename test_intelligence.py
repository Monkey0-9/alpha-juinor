#!/usr/bin/env python3
"""Test intelligence module imports."""

import traceback
import sys

modules = [
    "intelligence.elite_brain",
    "intelligence.return_predictor",
    "intelligence.portfolio_optimizer",
    "intelligence.alpha_generator",
    "intelligence.regime_detector",
    "intelligence.risk_manager",
    "intelligence.strategic_reasoner",
    "intelligence.multi_agent_ensemble",
    "intelligence.neural_predictor",
    "intelligence.master_controller",
    "intelligence.ultimate_controller",
]

print("=" * 60)
print("Testing Intelligence Module Imports")
print("=" * 60)

success = 0
failed = 0

for module in modules:
    try:
        __import__(module)
        print(f"✓ {module}")
        success += 1
    except Exception as e:
        print(f"✗ {module}")
        print(f"  Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed += 1
        break  # Stop at first error for clarity

print("=" * 60)
print(f"Success: {success}, Failed: {failed}")
