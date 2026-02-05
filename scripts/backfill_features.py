"""
Institutional Feature Backfill Script
Computes and stores features for all ACTIVE symbols in the universe.
"""

import json
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from mini_quant_fund.intelligence.feature_store import FeatureStore
from utils.logging_config import setup_logging

logger = setup_logging("FEATURE_BACKFILL", log_dir="runtime/logs")


def main():
    db = DatabaseManager()
    feature_store = FeatureStore()

    # 1. Get active tickers from config
    try:
        with open("configs/universe.json", "r") as f:
            universe = json.load(f)
            tickers = universe.get("active_tickers", [])
    except Exception as e:
        logger.error(f"Failed to load universe: {e}")
        return

    logger.info(f"Starting feature backfill for {len(tickers)} symbols...")

    # 2. Get existing feature coverage (since table was nuked, this is 100% missing)
    missing_tickers = list(tickers)
    logger.info(f"Detected {len(missing_tickers)} symbols missing features.")

    # 3. Load 252 days of data and compute
    success_count = 0
    for symbol in missing_tickers:
        try:
            # Query price history
            conn = db.adapter._get_connection()
            query = f"SELECT * FROM price_history WHERE symbol='{symbol}' ORDER BY date DESC LIMIT 1000"
            df = pd.read_sql_query(query, conn)

            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol} (rows: {len(df)})")
                continue

            # Rename columns to match expected format
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True
            )

            # Convert date to index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # Compute and store
            if feature_store.compute_and_store(symbol, df):
                success_count += 1
                logger.info(f"[{success_count}/{len(missing_tickers)}] Backfilled: {symbol}")
            else:
                logger.error(f"Failed to compute features for {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    logger.info(f"Backfill complete: {success_count}/{len(missing_tickers)} processed.")


if __name__ == "__main__":
    main()
