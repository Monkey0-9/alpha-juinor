
import sys
import traceback

print("Checking Elite Imports...")

try:
    from intelligence.elite_monte_carlo import get_elite_mc_predictor
    print("SUCCESS: elite_monte_carlo imported")
except ImportError:
    print("FAILURE: elite_monte_carlo failed to import")
    traceback.print_exc()
except Exception as e:
    print(f"ERROR: elite_monte_carlo error: {e}")
    traceback.print_exc()

try:
    from intelligence.elite_statistics import TimeSeriesStats
    print("SUCCESS: elite_statistics imported")
except ImportError:
    print("FAILURE: elite_statistics failed to import")
    traceback.print_exc()
except Exception as e:
    print(f"ERROR: elite_statistics error: {e}")
    traceback.print_exc()

import strategies.monte_carlo_mean_reversion as mr

print(f"Strategy ELITE_AVAILABLE: {mr.ELITE_AVAILABLE}")
