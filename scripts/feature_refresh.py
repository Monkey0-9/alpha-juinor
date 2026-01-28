
import sys
import os
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from mini_quant_fund.intelligence.feature_store import FeatureStore
from data.collectors.data_router import DataRouter

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FEATURE_REFRESH")

def refresh_features(symbols_limit=None, workers=4):
    """
    Refresh features for all/some symbols.
    """
    db = DatabaseManager()
    store = FeatureStore()

    # 1. Get Active Universe
    active_symbols = db.get_active_symbols()

    if not active_symbols:
        logger.warning("No ACTIVE symbols found in symbol_governance. Falling back to universe.json if needed or exit.")
        # Fallback to universe.json if DB is empty (bootstrapping)
        import json
        try:
             with open("configs/universe.json", "r") as f:
                  active_symbols = json.load(f).get("active_tickers", [])
        except:
             pass

    if not active_symbols:
        logger.error("No symbols to process.")
        return

    if symbols_limit:
        active_symbols = active_symbols[:symbols_limit]

    logger.info(f"Starting feature refresh for {len(active_symbols)} symbols with {workers} workers.")

    success_count = 0
    fail_count = 0

    def process_symbol(symbol):
        try:
            # Get daily price history (last 500 rows approx 2 years)
            df = db.get_daily_prices(symbol, limit=750)
            if df.empty or len(df) < 60:
                return False, f"Insufficient data: {len(df)}"

            # Compute and store
            ok = store.compute_and_store(symbol, df)
            if ok:
                return True, "OK"
            else:
                return False, "Compute/Validation Failed"
        except Exception as e:
            return False, str(e)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_symbol = {executor.submit(process_symbol, sym): sym for sym in active_symbols}

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                success, msg = future.result()
                if success:
                    success_count += 1
                    logger.info(f"[SUCCESS] {symbol}")
                else:
                    fail_count += 1
                    logger.warning(f"[FAIL] {symbol}: {msg}")
            except Exception as e:
                fail_count += 1
                logger.error(f"[ERROR] {symbol}: {e}")

    logger.info(f"Feature Refresh Complete. Success: {success_count}, Failed: {fail_count}")

    # Write summary to DB audit?
    # Not strictly required but good for governance.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Force refresh of features")
    parser.add_argument("--limit", type=int, help="Limit number of symbols", default=None)
    parser.add_argument("--workers", type=int, help="Parallel workers", default=4)
    args = parser.parse_args()

    refresh_features(symbols_limit=args.limit, workers=args.workers)
