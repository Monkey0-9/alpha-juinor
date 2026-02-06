import argparse
import logging

import numpy as np
import pandas as pd

from backtest.backtester import run_backtest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BACKTESTER_CLI")

def main():
    parser = argparse.ArgumentParser(description="Run Backtest")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")
    parser.add_argument("--compare", type=str, help="Baseline to compare against")
    args = parser.parse_args()

    logger.info(f"Running backtest for {args.strategy} over {args.days} days...")

    # Mock data generation for demo purposes (since we don't have easy DB access in this script yet)
    # In production, this would load from DB
    dates = pd.date_range(end=pd.Timestamp.now(), periods=args.days)
    prices = pd.Series(np.cumsum(np.random.randn(args.days)) + 100, index=dates)

    # Mock permissions (signals)
    # 1 = Invest, 0 = Cash
    permissions = pd.Series(np.random.choice([0, 1], size=args.days), index=dates)

    # Run Strategy Backtest
    results = run_backtest(prices, permissions)
    final_equity = results["equity"].iloc[-1]
    total_return = (final_equity - 1_000_000) / 1_000_000 * 100

    print(f"\n--- Results: {args.strategy} ---")
    print(f"Final Equity: ${final_equity:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {results['net_return'].mean() / results['net_return'].std() * np.sqrt(252):.2f}")

    # Run Baseline (if requested)
    if args.compare:
        print(f"\n--- Comparing with {args.compare} ---")
        # specific logic for legacy check
        legacy_permissions = pd.Series(np.ones(args.days), index=dates) # Buy and Hold
        base_results = run_backtest(prices, legacy_permissions)
        base_return = (base_results["equity"].iloc[-1] - 1_000_000) / 1_000_000 * 100
        print(f"Baseline Return: {base_return:.2f}%")
        print(f"Outperformance: {total_return - base_return:.2f}%")

if __name__ == "__main__":
    main()
