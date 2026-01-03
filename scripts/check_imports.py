
import sys
import os
sys.path.append(os.getcwd())
import traceback

print("Checking imports...")

try:
    print("1. Importing strategies.features...")
    from strategies.features import FeatureEngineer
    print("   Success.")
except ImportError:
    print("   FAILED.")
    traceback.print_exc()

try:
    print("2. Importing strategies.ml_alpha...")
    from strategies.ml_alpha import MLAlpha
    print("   Success.")
except ImportError:
    print("   FAILED.")
    traceback.print_exc()

try:
    print("3. Importing portfolio.optimizer...")
    from portfolio.optimizer import MeanVarianceOptimizer
    print("   Success.")
except ImportError:
    print("   FAILED.")
    traceback.print_exc()

try:
    print("4. Importing risk.factor_model...")
    from risk.factor_model import StatisticalRiskModel
    print("   Success.")
except ImportError:
    print("   FAILED.")
    traceback.print_exc()

print("Done.")
