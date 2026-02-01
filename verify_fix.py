import sys
import os
import pandas as pd
import numpy as np

print("1. Verifying Contracts...")
try:
    from contracts import decision_enum, AlphaDistribution
    print("   contracts imported successfully.")
except ImportError as e:
    print(f"   FAILED: {e}")

print("\n2. Verifying Risk Quantum...")
try:
    from risk.quantum.entanglement_detector import build_entanglement_matrix, entanglement_indices
    df = pd.DataFrame(np.random.normal(0,1, (100, 5)))
    e = build_entanglement_matrix(df)
    print("   Entanglement Detector exposed functions found.")
except ImportError as e:
    print(f"   FAILED: {e}")

print("\n3. Verifying ML Alpha...")
try:
    from alpha_families.ml_alpha import MLAlpha
    alpha = MLAlpha()
    ready, reasons = alpha.ml_training_ready(pd.DataFrame(index=range(6000)))
    print(f"   MLAlpha.ml_training_ready check: {ready}")
    if ready:
        print("   MLAlpha fixed.")
    else:
        print(f"   MLAlpha not ready? {reasons}")
except Exception as e:
    print(f"   FAILED: {e}")

print("\n4. Verifying Portfolio Optimizer...")
try:
    from portfolio.optimizer import PortfolioOptimizer
    opt = PortfolioOptimizer()
    print("   PortfolioOptimizer class found.")
except ImportError as e:
    print(f"   FAILED: {e}")

print("\nDone.")
