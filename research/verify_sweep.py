"""
Verify Hypothesis Sweep
=======================
Tests the automated discovery pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.hypothesis_manager import HypothesisManager
from research.rapid_research_backtester import RapidResearchBacktester
from research.sweep_runner import SweepRunner

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    print("Generating synthetic trending data...")
    # Create valid inputs for features (RSI needs 14 bars, SMA 50)
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')

    # Create a strong trend so at least Momentum features show positive Sharpe
    price = np.linspace(100, 200, 200) + np.random.normal(0, 2, 200)
    market_data = pd.DataFrame({'Close': price}, index=dates)

    print("Initializing Managers...")
    manager = HypothesisManager()
    backtester = RapidResearchBacktester()
    sweeper = SweepRunner(manager, backtester)

    print("Running Sweep...")
    sweeper.run_sweep(market_data)

    print("Checking Database for discoveries...")
    df = manager.list_hypotheses()
    print(f"Total Hypotheses Found: {len(df)}")

    if len(df) > 0:
        print("Latest Discoveries:")
        print(df[['name', 'metrics']].tail())
        print("SUCCESS: Automated sweep discovered strategies.")
    else:
        print("WARNING: No strategies passed filter (Sharpe > 0).")
        # Still a success of the pipeline mechanics, but let's check input if it fails often.

if __name__ == "__main__":
    main()
