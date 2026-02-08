"""
Verification script for Elite Monte Carlo Mean Reversion Strategy.
"""

import logging

import numpy as np
import pandas as pd

from strategies.monte_carlo_mean_reversion import MonteCarloMeanReversionStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler("verify.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger("VERIFY")


def print_log(msg):
    logger.info(msg)
    print(msg)


def run_verification():
    print_log("--- STARTING VERIFICATION ---")

    # 1. Generate Synthetic Data (Mean Reverting)
    np.random.seed(42)
    n_days = 200
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days)

    # Simulate a mean-reverting process (OU)
    prices = [100.0]
    mu = 100.0
    theta = 0.1
    sigma = 0.5  # Reduced volatility to avoid CRISIS regime detection

    for _ in range(1, n_days):
        dx = theta * (mu - prices[-1]) + sigma * np.random.normal()
        prices.append(prices[-1] + dx)

    price_series = pd.Series(prices, index=dates)

    # Force a dip to trigger BUY
    price_series.iloc[-1] = 95.0  # Moderate drop, should be significant for low vol
    price_series.iloc[-2] = 98.0
    price_series.iloc[-3] = 99.0

    print_log(f"Current Price: {price_series.iloc[-1]:.2f}")

    # 2. Run Strategy
    strategy = MonteCarloMeanReversionStrategy()
    print_log("\n--- INSTANTIATED STRATEGY ---")

    signal = strategy.generate_signal("TEST_SYM", price_series)
    print_log("\n--- SIGNAL GENERATED ---")

    # 3. Output Results
    print_log(strategy.explain_decision(signal))

    print_log("-" * 40)
    print_log(f"Signal Strength: {signal.strength:.3f}")
    print_log(f"Confidence:     {signal.confidence:.3f}")
    print_log(f"Signal Type:    {signal.metadata.get('signal_type', 'N/A')}")
    print_log(f"RSI:            {signal.metadata.get('rsi', 0):.1f}")
    print_log(f"Regime:         {signal.metadata.get('regime', 'N/A')}")
    print_log(f"Hurst:          {signal.metadata.get('hurst_exponent', 0):.3f}")
    print_log(f"Kelly:          {signal.metadata.get('kelly_fraction', 0):.3f}")
    print_log(f"Mispricing:     {signal.metadata.get('mispricing_score', 0):.3f}")

    # Debug info
    fv = signal.metadata.get("fair_value_range", (0, 0))
    print_log(f"FV Range:       {fv[0]:.2f} - {fv[1]:.2f}")
    print_log(f"Current Price:  {price_series.iloc[-1]:.2f}")

    # 4. Explicit Method Testing
    print_log("\n--- DIAGNOSTICS ---")
    try:
        stats = strategy._get_advanced_stats(price_series)
        print_log(
            f"Direct Stats Call: Hurst={stats['hurst']:.3f}, Rw={stats['is_random_walk']}"
        )
    except Exception as e:
        print_log(f"ERROR: _get_advanced_stats failed: {e}")
        import traceback

        traceback.print_exc()

    try:
        metrics = strategy._get_prediction_metrics("TEST_SYM", price_series)
        print_log(
            f"Direct Pred Call:  Regime={metrics['regime']}, Conf={metrics['model_confidence']}"
        )
    except Exception as e:
        print_log(f"ERROR: _get_prediction_metrics failed: {e}")
        import traceback

        traceback.print_exc()

    if signal.strength > 0:
        print_log("\nSUCCESS: Buy signal generated as expected for undervalued asset.")
    else:
        print_log("\nWARNING: No positive signal generated.")


if __name__ == "__main__":
    run_verification()
