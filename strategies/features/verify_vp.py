"""
Verify Volume Profile
=====================
Tests calculation of POC, VAH, VAL on synthetic Gaussian data.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from strategies.features.volume_profile_analyzer import VolumeProfileAnalyzer

def main():
    print("Generating synthetic Gaussian market data...")
    # Simulate price hovering around 100 with normal distribution
    np.random.seed(42)
    n_points = 1000
    prices = np.random.normal(100.0, 2.0, n_points)
    volumes = np.random.randint(100, 1000, n_points)

    # Create Panel-like structure using MultiIndex
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='T')

    # Construct MultiIndex DataFrame
    mi = pd.MultiIndex.from_product([['SYTH'], dates], names=['Symbol', 'Date'])

    # Align data length
    # Wait, simple way: dict of dfs
    # Or strict MultiIndex:
    # 1000 rows.
    df = pd.DataFrame({'Close': prices, 'Volume': volumes}, index=dates)

    # To mimic structure passed to strategies:
    # columns could be MultiIndex: (Symbol, Field) e.g. ('SYTH', 'Close')
    full_df = pd.DataFrame({
        ('SYTH', 'Close'): prices,
        ('SYTH', 'Volume'): volumes
    }, index=dates)

    print("Initializing Volume Profile Analyzer...")
    vp = VolumeProfileAnalyzer(n_bins=50)
    vp.update(full_df)

    levels = vp.get_key_levels('SYTH')
    nodes = vp.get_nodes('SYTH')

    print(f"Results for SYTH:")
    print(f"POC (Expected ~100.0): {levels.get('POC'):.2f}")
    print(f"VAH: {levels.get('VAH'):.2f}")
    print(f"VAL: {levels.get('VAL'):.2f}")
    print(f"HVN Count: {len(nodes.get('HVN'))}")

    # Validation Check
    if 98.0 < levels.get('POC') < 102.0:
        print("SUCCESS: POC is centered correctly.")
        sys.exit(0)
    else:
        print("FAILURE: POC measurement aberrant.")
        sys.exit(1)

if __name__ == "__main__":
    main()
