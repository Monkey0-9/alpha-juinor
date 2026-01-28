"""
scripts/compare_fills.py

Compares simulated fills vs real execution fills (if available).
"""

import argparse
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FILL_COMPARE")

def compare_fills(start: str, end: str, output: str):
    logger.info("Comparing Fills...")

    # 1. Load Simulated Fills (from backtest/paper logs)
    # 2. Load Real/Paper Broker Fills (from broker export or runtime DB)

    # Placeholder: Generate empty CSV structure
    df = pd.DataFrame(columns=[
        "Symbol", "Side", "Qty",
        "SimPric", "RealPrice", "Slippage", "Shortfall", "Timestamp"
    ])

    df.to_csv(output, index=False)
    logger.info(f"Comparison report saved to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="-30d")
    parser.add_argument("--end", default="now")
    parser.add_argument("--output", default="fill_comparison.csv")
    args = parser.parse_args()

    compare_fills(args.start, args.end, args.output)
