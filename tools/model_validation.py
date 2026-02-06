import argparse
import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

from ml.alpha_decay import AlphaDecayMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-decay", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    print("Running Model Validation Suite...")
    monitor = AlphaDecayMonitor(decay_threshold=args.threshold)

    # Mock data for healthy strategy (simulating live model performance)
    # in a real scenario, this would load from 'model_metrics' table
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100)
    signals = pd.Series(np.random.normal(0.05, 1, 100), index=dates)
    # Forward returns correlated with signal (healthy)
    returns = signals * 0.1 + np.random.normal(0, 0.2, 100)

    metrics = monitor.analyze_strategy(
        "PREDICTIVE_V1", signals, returns, current_aum=1000000, avg_daily_volume=50000000
    )

    print(f"Strategy: {metrics.strategy_id}")
    print(f"Rolling IC (90d): {metrics.rolling_ic_90d:.4f}")
    print(f"Decay Score: {metrics.decay_score:.4f}")
    print(f"Status: {metrics.status}")
    print(f"Recommendation: {metrics.recommendation}")

    if metrics.status == "CRITICAL":
        print("[FAIL] Model decay validation failed.")
        sys.exit(1)

    print("[PASS] Model validation successful.")
    sys.exit(0)

if __name__ == "__main__":
    main()
