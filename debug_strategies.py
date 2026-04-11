import sys
import os

def test_import_strategies():
    import sys
    print(f"\nPYTHONPATH: {sys.path}")
    import strategies
    print(f"Successfully imported strategies: {strategies}")
    import strategies.monte_carlo_mean_reversion
    print("Successfully imported strategies.monte_carlo_mean_reversion")
