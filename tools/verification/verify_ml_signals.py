"""
Verify ML Alpha Signal Generation.
Runbook Item: python verify_ml_signals.py
"""
import sys
import logging
import pandas as pd
from alpha_families.ml_alpha import MLAlpha
from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("VERIFY_ML")

def main():
    logger.info("Initializing MLAlpha...")
    alpha = MLAlpha()
    db = DatabaseManager()

    test_symbols = ["AAPL", "ETH-USD"]

    logger.info(f"Testing Signal Generation for: {test_symbols}")

    for symbol in test_symbols:
        logger.info(f"--- Testing {symbol} ---")

        # 1. Fetch Data manually to ensure it exists
        df = db.get_daily_prices(symbol, limit=252)
        if df.empty:
            logger.error(f"FAIL: No data for {symbol}")
            continue

        logger.info(f"Data Loaded: {len(df)} rows")

        # 2. Generate Signal
        # We need to mock the market_data_map structure expected by generate_signal
        # Expected: {ticker: df} where df has OHLCV columns

        market_data_map = {symbol: df}

        try:
            signal = alpha.generate_signal(symbol, market_data_map)
            logger.info(f"Signal Result: {signal}")

            if signal == 0.0:
                 logger.warning(f"NEUTRAL SIGNAL (0.0). Check model availability.")
                 # Investigate which model was used
                 cached = alpha._cached_models.get(symbol) or alpha._cached_models.get("GLOBAL") or alpha._cached_models.get("LEGACY_GLOBAL")
                 if cached:
                     logger.info(f"Model Used: {cached.get('status', 'Unknown')}")
                 else:
                     logger.error("No model was loaded!")
            else:
                logger.info("SUCCESS: Non-neutral signal generated.")

        except Exception as e:
            logger.error(f"CRITICAL ERROR generating signal: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
