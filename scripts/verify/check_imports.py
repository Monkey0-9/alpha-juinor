
import sys
import os
sys.path.append(os.getcwd())
import traceback

print("Checking imports...")

try:
    print("1. Importing strategies.features...")
    from mini_quant_fund.strategies.features import FeatureEngineer
    print("   Success.")
except ImportError:
    print("   FAILED.")
    traceback.print_exc()

try:
    print("2. Importing strategies.ml_alpha...")
    from mini_quant_fund.strategies.ml_alpha import MLAlpha
    print("   Success.")
except ImportError:
    print("   FAILED.")
    traceback.print_exc()

try:
    print("3. Importing portfolio.optimizer...")
    from mini_quant_fund.portfolio.optimizer import MeanVarianceOptimizer
    print("   Success.")
except ImportError:
    print("   FAILED.")
    traceback.print_exc()

try:
    print("4. Importing risk.factor_model...")
    from mini_quant_fund.risk.factor_model import StatisticalRiskModel
    print("   Success.")
except ImportError:
    print("   FAILED.")
    traceback.print_exc()

print("Done.")
