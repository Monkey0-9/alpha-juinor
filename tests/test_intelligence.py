#!/usr/bin/env python3
"""Test intelligence module imports."""

import traceback
import sys

modules = [
    "mini_quant_fund.intelligence.elite_brain",
    "mini_quant_fund.intelligence.return_predictor",
    "mini_quant_fund.intelligence.portfolio_optimizer",
    "mini_quant_fund.intelligence.alpha_generator",
    "mini_quant_fund.intelligence.regime_detector",
    "mini_quant_fund.intelligence.risk_manager",
    "mini_quant_fund.intelligence.strategic_reasoner",
    "mini_quant_fund.intelligence.multi_agent_ensemble",
    "mini_quant_fund.intelligence.neural_predictor",
    "mini_quant_fund.intelligence.master_controller",
    "mini_quant_fund.intelligence.ultimate_controller",
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
