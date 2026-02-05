import pandas as pd
import numpy as np
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RLDataPrep")

def generate_mock_strategy_returns(check_path=None):
    """
    Generate synthetic returns for strategies if real data missing.
    """
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="B")

    # Strategies
    strats = ["MeanReversion", "TrendFollowing", "NetworkAlpha"]

    data = {}
    for s in strats:
        # Random walk with slight positive bias (Sharpe ~1.0)
        daily_ret = np.random.normal(0.0005, 0.01, size=len(dates))
        data[s] = daily_ret

    df = pd.DataFrame(data, index=dates)

    # Save
    os.makedirs("data/training", exist_ok=True)
    out_path = "data/training/strategy_returns.csv"
    df.to_csv(out_path)
    logger.info(f"Generated synthetic training data at {out_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="synthetic", help="Source of returns")
    args = parser.parse_args()

    if args.source == "synthetic":
        generate_mock_strategy_returns()
    else:
        logger.info("Real data loading not yet implemented. Using synthetic.")
        generate_mock_strategy_returns()
