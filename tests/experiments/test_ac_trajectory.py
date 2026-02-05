"""
Verification script for Almgren-Chriss Optimizer.
"""

import pandas as pd
import numpy as np
from execution.strategies.almgren_chriss import AlmgrenChrissOptimizer, ACMeta

def verify_ac_trajectory():
    # Parameters for a typical institutional trade
    # Sell 1,000,000 shares over 1 day (6.5 hours)
    meta = ACMeta(
        total_shares=1000000,
        start_price=100.0,
        daily_volatility=2.0,   # High vol ($2 move)
        risk_aversion=1e-5,     # Typical institutional aversion
        permanent_impact=2.5e-7,
        temporary_impact=2.5e-6,
        time_horizon_days=1.0,
        interval_days=1.0/20.0  # 20 intervals
    )

    optimizer = AlmgrenChrissOptimizer(meta)
    trajectory = optimizer.compute_trajectory()

    print("\n=== Almgren-Chriss Optimization Results ===")
    print(f"Total Shares: {meta.total_shares}")
    print(f"Risk Aversion: {meta.risk_aversion}")
    print("-" * 40)
    print(trajectory[['time_offset', 'shares_held', 'trade_size']].head(10))
    print("...")
    print(trajectory[['time_offset', 'shares_held', 'trade_size']].tail(5))
    print("-" * 40)

    total_traded = trajectory['trade_size'].sum()
    print(f"Total Traded: {total_traded:,.0f} (Expected: {meta.total_shares:,.0f})")

    # Check if Front-Loaded
    first_half = trajectory.iloc[:10]['trade_size'].sum()
    second_half = trajectory.iloc[10:]['trade_size'].sum()

    print(f"First Half Vol: {first_half:,.0f}")
    print(f"Second Half Vol: {second_half:,.0f}")

    if first_half > second_half:
        print("✅ Trajectory is Front-Loaded (Correct behavior for Risk Averse)")
    else:
        print("⚠️ Trajectory is NOT Front-Loaded (Check params)")

if __name__ == "__main__":
    verify_ac_trajectory()
