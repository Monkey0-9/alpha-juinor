#!/usr/bin/env python3
"""Test each module individually."""
import sys
import traceback

files = [
    ("elite_brain.py", "intelligence/elite_brain.py"),
    ("return_predictor.py", "intelligence/return_predictor.py"),
    ("portfolio_optimizer.py", "intelligence/portfolio_optimizer.py"),
    ("alpha_generator.py", "intelligence/alpha_generator.py"),
    ("regime_detector.py", "intelligence/regime_detector.py"),
    ("risk_manager.py", "intelligence/risk_manager.py"),
    ("strategic_reasoner.py", "intelligence/strategic_reasoner.py"),
    ("multi_agent_ensemble.py", "intelligence/multi_agent_ensemble.py"),
    ("neural_predictor.py", "intelligence/neural_predictor.py"),
    ("master_controller.py", "intelligence/master_controller.py"),
    ("ultimate_controller.py", "intelligence/ultimate_controller.py"),
]

for name, path in files:
    try:
        with open(path, 'r') as f:
            code = f.read()
        exec(compile(code, path, 'exec'), {'__name__': '__main__'})
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
        traceback.print_exc()
        break
