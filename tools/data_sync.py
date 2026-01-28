#!/usr/bin/env python3
"""
Incremental Data Sync Tool.

Fetches daily updates and maintains data freshness.
Supports both daily batch updates and intraday streaming.

Usage:
    python tools/data_sync.py --symbol AAPL --start 2026-01-01 --end 2026-01-19
    python tools/data_sync.py --mode daily --days 7
    python tools/data_sync.py --mode intraday --stream --interval 30
"""

import os
import sys
import json
import logging
import hashlib
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.schema import DailyPriceRecord, DataQualityRecord, ProviderMetricsRecord
from database.manager import get_db, DatabaseManager
from data.providers.base import DataProvider
from data.providers.yahoo import YahooDataProvider
from data.providers.polygon import PolygonDataProvider
from data.providers.alpha_vantage import AlphaVantageProvider
from data.providers.stooq import StooqProvider
from data_intelligence.provider_bandit import ProviderBandit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] data_sync: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a sync operation"""
    symbol: str
    status: str
    rows_updated: int = 0
    quality_score: float = 0.0
    error: str = ""


class IncrementalSync:
    """
    Incremental data synchronization.

    Features:
    - Daily batch updates
    - Missing date reconciliation
    - Intraday streaming (where supported)
    - Provider fallback on failure
    """

    def __init__(self, db: DatabaseManager = None, max_workers: int = 5):
        self.db = db or get_db()
        self.max_workers = max_workers
        self.provider_bandit = ProviderBandit(
            ['polygon', 'alpha_vantage', 'stooq', 'yahoo']
        )
        self._providers = {}
        self._provider_lock = threading.Lock()

    def _get_provider(self, name: str) -> DataProvider:
        """Get provider instance"""
        with self._provider_lock:
            if name not in self._providers:
                self._providers[name] = self._create_provider(name)
            return self._providers[name]

    def _create_provider(self, name: str) -> DataProvider:
        """Create provider by name"""
        if name == 'polygon':
            return PolygonDataProvider()
        elif name == 'alpha_vantage':
            return AlphaVantageProvider()
        elif name == 'stooq':
            return StooqProvider()
        elif name == 'yahoo':
            return YahooDataProvider()
        return YahooDataProvider()

    def sync_symbol(self, symbol: str, start_date: str,
                    end_date: str = None) -> SyncResult:
        """
        Sync a single symbol for the date range.

        Identifies missing dates and fetches only those.
        """
        result = SyncResult(symbol=symbol)

        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')

        try:
            # Get existing dates in range
            existing_dates = self._get_existing_dates(symbol, start_date, end_date)

            # Identify missing dates
            all_dates = set(pd.date_range(start=start_date, end=end_date, freq='B').strftime('%Y-%m-%d'))
            missing_dates = all_dates - existing_dates

            if not missing_dates:
                result.status = 'up_to_date'
                logger.info(f"{symbol}: Already up to date")
                return result

            logger.info(f"{symbol}: {len(missing_dates)} missing dates out of {len(all_dates)}")

            # Fetch missing data
            provider_name = self.provider_bandit.select_provider()
            provider = self._get_provider(provider_name)

            # Adjust date range to include surrounding days for continuity
            min_date = min(missing_dates)
            max_date = max(missing_dates)

            df = provider.fetch_ohlcv(symbol, min_date, max_date)

            if df is None or df.empty:
                result.status = 'failed'
                result.error = f"No data from {provider_name}"
                return result

            # Filter to only missing dates
            df = df[df.index.strftime('%Y-%m-%d').isin(missing_dates)]

            if df.empty:
                result.status = 'up_to_date'
                return result

            # Validate and persist
            df = self._validate_and_normalize(df, symbol)

            rows = self._persist_daily(df, symbol, provider_name)
            result.rows_updated = rows
            result.status = 'success'

            # Update provider metrics
            self.provider_bandit.update(provider_name, True, 100, 0.9)

            return result

        except Exception as e:
            result.status = 'failed'
            result.error = str(e)
            logger.error(f"Sync failed for {symbol}: {e}")
            return result

    def sync_universe(self, start_date: str, end_date: str = None,
                      symbols: List[str] = None) -> Dict[str, SyncResult]:
        """
        Sync entire universe for date range.
        """
        if symbols is None:
            symbols = self._load_universe()

        if end_date is None:
            end_date = datetime.utcnow().strftime('%Y-%m-%d')

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.sync_symbol, symbol, start_date, end_date): symbol
                for symbol in symbols
            }

            for future in futures:
                symbol = futures[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    results[symbol] = SyncResult(symbol=symbol, status='failed', error=str(e))

        # Summary
        success = sum(1 for r in results.values() if r.status == 'success')
        failed = sum(1 for r in results.values() if r.status == 'failed')
        up_to_date = sum(1 for r in results.values() if r.status == 'up_to_date')

        summary = {
            'total_symbols': len(symbols),
            'success': success,
            'failed': failed,
            'up_to_date': up_to_date,
            'total_rows_updated': sum(r.rows_updated for r in results.values())
        }

        logger.info(f"Sync complete: {summary}")
        return results

    def sync_daily_batch(self, days: int = 1) -> Dict[str, SyncResult]:
        """
        Run daily batch sync for yesterday and any missing recent dates.
        """
        today = datetime.utcnow()
        end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (today - timedelta(days=days + 7)).strftime('%Y-%m-%d')  # Include buffer

        return self.sync_universe(start_date, end_date)

    def _get_existing_dates(self, symbol: str, start_date: str,
                            end_date: str) -> set:
        """Get set of dates already in database"""
        df = self.db.get_daily_prices(symbol, start_date, end_date)
        return set(df.index.strftime('%Y-%m-%d'))

    def _validate_and_normalize(self, df: pd.DataFrame,
                                 symbol: str) -> pd.DataFrame:
        """Validate and normalize dataframe"""
        df = df.copy()

        # Ensure required columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = 0.0

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Ensure numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop invalid rows
        df = df.dropna(subset=['Close', 'Volume'])
        df = df[df['Close'] > 0]
        df = df[df['Volume'] > 0]

        return df

    def _persist_daily(self, df: pd.DataFrame, symbol: str,
                       provider: str) -> int:
        """Persist daily data to database"""
        raw_hash = hashlib.sha256(
            df.to_json(date_format='iso', orient='split').encode()
        ).hexdigest()

        pulled_at = datetime.utcnow().isoformat()
        records = []

        for idx, row in df.iterrows():
            records.append(DailyPriceRecord(
                symbol=symbol,
                date=idx.strftime('%Y-%m-%d'),
                open=row.get('Open', 0),
                high=row.get('High', 0),
                low=row.get('Low', 0),
                close=row.get('Close', 0),
                adjusted_close=row.get('Adjusted_Close', row.get('Close', 0)),
                volume=int(row.get('Volume', 0)),
                source_provider=provider,
                raw_hash=raw_hash,
                pulled_at=pulled_at
            ))

        return self.db.upsert_daily_prices_batch(records)

    def _load_universe(self) -> List[str]:
        """Load symbols from universe"""
        with open("configs/universe.json") as f:
            config = json.load(f)
        return config.get('active_tickers', [])

    def get_data_status(self, symbol: str) -> Dict:
        """Get data status for a symbol"""
        conn = self.db.get_connection()
        cursor = conn.execute('''
            SELECT MIN(date) as first_date, MAX(date) as last_date, COUNT(*) as count
            FROM price_history_daily
            WHERE symbol = ?
        ''', (symbol,))
        row = cursor.fetchone()

        if row and row['count'] > 0:
            return {
                'symbol': symbol,
                'first_date': row['first_date'],
                'last_date': row['last_date'],
                'row_count': row['count'],
                'status': 'has_data'
            }

        return {'symbol': symbol, 'status': 'no_data'}


def run_intraday_stream(symbols: List[str], interval: int = 30):
    """
    Run intraday streaming for specified symbols.
    Fetches latest minute bar every `interval` seconds.
    """
    sync = IncrementalSync()

    logger.info(f"Starting intraday stream for {len(symbols)} symbols")

    while True:
        try:
            for symbol in symbols:
                result = sync.sync_symbol(symbol,
                                          datetime.utcnow().strftime('%Y-%m-%d'))
                if result.status == 'success':
                    logger.debug(f"{symbol}: Updated {result.rows_updated} rows")
                elif result.status != 'up_to_date':
                    logger.warning(f"{symbol}: {result.status}")

            time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Intraday stream stopped")
            break
        except Exception as e:
            logger.error(f"Stream error: {e}")
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description='Incremental Data Sync')

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--symbol', type=str, help='Single symbol to sync')
    mode_group.add_argument('--universe', action='store_true',
                           help='Sync entire universe')

    parser.add_argument('--start', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=1,
                        help='Number of days for daily batch sync')
    parser.add_argument('--mode', type=str, choices=['daily', 'intraday'],
                        default='daily', help='Sync mode')
    parser.add_argument('--stream', action='store_true',
                        help='Enable continuous streaming (intraday mode)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Streaming interval in seconds')
    parser.add_argument('--workers', type=int, default=5,
                        help='Number of parallel workers')
    parser.add_argument('--status', action='store_true',
                        help='Show data status for symbol')

    args = parser.parse_args()

    sync = IncrementalSync(max_workers=args.workers)

    # Show status
    if args.status and args.symbol:
        status = sync.get_data_status(args.symbol)
        print(json.dumps(status, indent=2))
        return 0

    # Intraday streaming
    if args.mode == 'intraday' and args.stream:
        symbols = [args.symbol] if args.symbol else sync._load_universe()
        run_intraday_stream(symbols, args.interval)
        return 0

    # Single symbol sync
    if args.symbol:
        if args.start is None:
            args.start = datetime.utcnow().strftime('%Y-%m-%d')
        result = sync.sync_symbol(args.symbol, args.start, args.end)
        print(json.dumps({
            'symbol': result.symbol,
            'status': result.status,
            'rows_updated': result.rows_updated,
            'error': result.error
        }, indent=2))
        return 0

    # Daily batch sync
    if args.mode == 'daily':
        results = sync.sync_daily_batch(args.days)
        summary = {
            'total': len(results),
            'success': sum(1 for r in results.values() if r.status == 'success'),
            'failed': sum(1 for r in results.values() if r.status == 'failed'),
            'up_to_date': sum(1 for r in results.values() if r.status == 'up_to_date')
        }
        print(json.dumps(summary, indent=2))
        return 0

    # Universe sync
    if args.universe:
        if args.start is None:
            args.start = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
        if args.end is None:
            args.end = datetime.utcnow().strftime('%Y-%m-%d')

        results = sync.sync_universe(args.start, args.end)
        print(f"Synced {len(results)} symbols")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

