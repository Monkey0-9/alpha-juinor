#!/usr/bin/env python3
"""
DB Backfill & Repair Tool

Responsibility:
1. Detect gaps in price_history.
2. Re-ingest specific slices.
3. Compare data across providers.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from ingest_history import InstitutionalIngestionAgent

logging.basicConfig(level=logging.INFO, format='[REPAIR] %(message)s')
logger = logging.getLogger("DB_REPAIR")

def detect_gaps(symbol: str, days: int = 1825):
    """Detect gaps in price history for a symbol."""
    db = DatabaseManager()
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = db.get_daily_prices(symbol, start_date=start_date)

    if df.empty:
        logger.warning(f"No data found for {symbol} in last {days} days.")
        return []

    # Simple gap detection (missing dates)
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B') # Business days
    missing = all_dates.difference(df.index)

    if not missing.empty:
        logger.info(f"Found {len(missing)} missing business days for {symbol}")
        return missing.tolist()
    return []

def repair_symbol(symbol: str):
    """Trigger repair for a single symbol."""
    logger.info(f"Triggering repair for {symbol}...")
    agent = InstitutionalIngestionAgent()
    res = agent.process_symbol(symbol)
    logger.info(f"Repair result for {symbol}: {res}")

def main():
    parser = argparse.ArgumentParser(description="Institutional DB Backfill & Repair Tool")
    parser.add_argument("--symbol", type=str, help="Specific symbol to repair")
    parser.add_argument("--detect-only", action="store_true", help="Only detect gaps, do not repair")
    parser.add_argument("--all-universe", action="store_true", help="Scan everything in universe")

    args = parser.parse_args()

    if args.symbol:
        gaps = detect_gaps(args.symbol)
        if gaps and not args.detect_only:
            repair_symbol(args.symbol)
    elif args.all_universe:
        import json
        with open("configs/universe.json", "r") as f:
            universe = json.load(f)
        for tk in universe.get("active_tickers", []):
            gaps = detect_gaps(tk)
            if gaps and not args.detect_only:
                repair_symbol(tk)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
