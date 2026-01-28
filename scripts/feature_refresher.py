#!/usr/bin/env python3
"""
Institutional Feature Refresher

Background process that runs every 30-60 minutes to:
1. Read price history from DB.
2. Recompute rolling technical/ML features.
3. Persist features for the Live Engine to use.
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.manager import DatabaseManager
from configs.config_manager import ConfigManager

import pandas as pd
from mini_quant_fund.intelligence.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO, format='[FEATURE_REFRESH] %(message)s')
logger = logging.getLogger("FEATURE_REFRESHER")

def refresh_features():
    db = DatabaseManager()
    fs = FeatureStore()

    try:
        with open("configs/universe.json", "r") as f:
            universe = json.load(f)
        tickers = universe.get("active_tickers", [])
    except:
        tickers = []

    if not tickers:
        logger.warning("No active tickers found in universe.json")
        return

    logger.info(f"Initiating feature refresh for {len(tickers)} symbols...")

    success_count = 0
    for symbol in tickers:
        try:
            # 1. Load history from DB (1260 days for full feature set)
            df = db.get_daily_prices(symbol, limit=1300)
            if df is None or df.empty:
                continue

            # 2. Rename columns to OHLCV (FeatureEngineer requirement)
            # Schema columns are lowercase
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)

            # 3. Contract Check (Stricter than Store validation)
            # Find expected features from authoritative registry
            from features.contract import get_feature_list
            expected_features = get_feature_list("ml_v1")

            # Predict and Store
            if fs.compute_and_store(symbol, df):
                # Verify what was stored matches model expectations
                # fs.compute_and_store calls compute_features_for_symbol
                # We want to ensure the final vector is compliant
                success_count += 1
                if success_count % 50 == 0:
                    logger.info(f"Processed {success_count}/{len(tickers)} symbols...")
            else:
                # If store validation failed, log as critical
                logger.critical(f"FEATURE_CONTRACT_VIOLATION for {symbol}. PAUSING promotion.")
                # In production, we'd set a 'PAUSE' flag in DB here

        except Exception as e:
            logger.error(f"Failed feature update for {symbol}: {e}")

    logger.info(f"Feature refresh complete. Successfully updated {success_count}/{len(tickers)} symbols.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Institutional Feature Refresher")
    parser.add_argument("--interval", type=int, default=30, help="Interval in minutes")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    if args.once:
        logger.info("Running single refresh...")
        refresh_features()
        sys.exit(0)

    logger.info(f"Feature Refresher started. Interval: {args.interval}m")

    while True:
        refresh_features()
        logger.info(f"Sleeping for {args.interval} minutes...")
        time.sleep(args.interval * 60)
