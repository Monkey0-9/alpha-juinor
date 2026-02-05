#!/usr/bin/env python3
"""
BACKTEST DATA INGESTION
=======================

Ingests 15+ years of historical data for institutional validation.
Uses Yahoo Finance for extended history.
"""

import sys
import os
import logging
from datetime import datetime

# Project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("BACKTEST_INGEST")

# Universe for backtesting (liquid, long-history assets)
BACKTEST_UNIVERSE = [
    # Major ETFs (high liquidity, long history)
    "SPY", "QQQ", "IWM", "DIA", "EEM", "EFA",
    "TLT", "IEF", "LQD", "HYG", "AGG", "BND",
    "GLD", "SLV", "USO", "UNG",
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "VNQ", "VWO", "VEA", "VTI", "VTV", "VUG",
    # Major stocks (long history)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS",
    "JNJ", "PFE", "UNH", "MRK", "ABBV",
    "XOM", "CVX", "COP",
    "WMT", "COST", "TGT", "HD", "LOW",
    "PG", "KO", "PEP", "MCD", "NKE",
]

START_DATE = "2007-01-01"  # Covers 2008 crisis


def ingest_backtest_data():
    """Ingest historical data for backtesting."""
    from data.storage.core import DataStore

    logger.info("=" * 60)
    logger.info("BACKTEST DATA INGESTION")
    logger.info(f"Universe: {len(BACKTEST_UNIVERSE)} symbols")
    logger.info(f"Start Date: {START_DATE}")
    logger.info("=" * 60)

    try:
        import yfinance as yf
        logger.info("Using yfinance for data")
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return 1

    store = DataStore()
    success = 0
    failed = 0

    for i, symbol in enumerate(BACKTEST_UNIVERSE, 1):
        try:
            logger.info(f"[{i}/{len(BACKTEST_UNIVERSE)}] Fetching {symbol}...")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=START_DATE, end=datetime.now().strftime("%Y-%m-%d"))

            if df.empty:
                logger.warning(f"  No data for {symbol}")
                failed += 1
                continue

            # Standardize columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Save
            store.save(symbol, df)

            years = len(df) / 252
            logger.info(f"  Saved {len(df)} bars (~{years:.1f} years)")
            success += 1

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed += 1

    logger.info("=" * 60)
    logger.info(f"COMPLETE: {success} success, {failed} failed")
    logger.info("=" * 60)

    return 0 if failed < len(BACKTEST_UNIVERSE) / 2 else 1


if __name__ == "__main__":
    sys.exit(ingest_backtest_data())
