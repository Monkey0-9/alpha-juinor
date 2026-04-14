import sys
import os

def test_import_strategies():
    import sys
    print(f"\nPYTHONPATH: {sys.path}")
    import mini_quant_fund.strategies
    print(f"Successfully imported strategies: {strategies}")
    import mini_quant_fund.strategies.monte_carlo_mean_reversion
    print("Successfully imported strategies.monte_carlo_mean_reversion")
