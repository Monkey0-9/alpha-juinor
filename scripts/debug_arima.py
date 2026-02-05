import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alpha_families.statistical_alpha import StatisticalAlpha

# from data.manager import DataManager # Removed as it might not exist or be needed for synthetic test


def debug_arima():
    print("Testing ARIMA Alpha...")
    alpha = StatisticalAlpha()

    # Load sample data for a few symbols (mocking or real if available)
    # We'll try to verify with a synthetic dataframe first, then real if possible
    print("Testing Synthetic Data...")
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    df = pd.DataFrame({
        'Close': np.random.randn(100).cumsum() + 100
    }, index=dates)

    print("Running ARIMA on synthetic data...")
    res = alpha.arima_safe_predict(df, symbol="SYNTHETIC")
    print(f"Synthetic Result: {res}")

    # If we have real data files, try to load one
    # Assuming parquet or similar structure, but DataManager handles it.
    # for now we stick to synthetic to verify the CODE logic vs DATA issue.

    # Let's try to simulate a short series failure
    print("\nTesting Short Series...")
    short_df = df.iloc[-10:]
    res_short = alpha.arima_safe_predict(short_df, symbol="SHORT_SYNTH")
    print(f"Short Series Result: {res_short}")

    # Test "Perfect" Linear data (should converge)
    print("\nTesting Linear Data...")
    linear_df = pd.DataFrame({
        'Close': np.linspace(100, 200, 100)
    }, index=dates)
    res_linear = alpha.arima_safe_predict(linear_df, symbol="LINEAR")
    print(f"Linear Result: {res_linear}")

if __name__ == "__main__":
    debug_arima()
