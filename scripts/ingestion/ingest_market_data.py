#!/usr/bin/env python3
"""
Market Data Ingestion Pipeline

Sole responsibility: FETCH, VALIDATE, and PERSIST the LAST 5 YEARS of historical market data
for EVERY symbol in the system universe.

This is NOT a trading system - it feeds downstream portfolio, risk, and execution engines.
"""

import os
import sys
import json
import logging
import hashlib
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import threading

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/ingestion_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
FIVE_YEARS_DAYS = 5 * 365  # Approximate
MIN_DATA_QUALITY = 0.6
DEFAULT_START_DATE = (datetime.utcnow() - timedelta(days=FIVE_YEARS_DAYS)).strftime('%Y-%m-%d')
DEFAULT_END_DATE = datetime.utcnow().strftime('%Y-%m-%d')


@dataclass
class IngestionResult:
    """Result of ingesting a single symbol"""
    symbol: str
    status: str  # SUCCESS, FAILED, INVALID_DATA
    data_quality_score: float = 0.0
    row_count: int = 0
    error_message: Optional[str] = None
    failed_checks: List[str] = field(default_factory=list)
    provider_used: str = ""
    start_date: str = ""
    end_date: str = ""


@dataclass
class IngestionSummary:
    """Summary of the entire ingestion run"""
    run_id: str
    total_symbols: int
    successful_fetches: int = 0
    failed_fetches: int = 0
    invalid_data_count: int = 0
    average_data_quality_score: float = 0.0
    symbols_failed: List[Dict[str, str]] = field(default_factory=list)
    symbols_invalid_data: List[Dict[str, Any]] = field(default_factory=list)
    provider_health_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    duration_seconds: float = 0.0
    start_time: str = ""
    end_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'total_symbols': self.total_symbols,
            'successful_fetches': self.successful_fetches,
            'failed_fetches': self.failed_fetches,
            'invalid_data_count': self.invalid_data_count,
            'average_data_quality_score': round(self.average_data_quality_score, 4),
            'symbols_failed': self.symbols_failed,
            'symbols_invalid_data': [
                {'symbol': s['symbol'], 'score': s['score'], 'checks': s.get('failed_checks', [])}
                for s in self.symbols_invalid_data
            ],
            'provider_health_stats': self.provider_health_stats,
            'duration_seconds': round(self.duration_seconds, 2),
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class DataQualityValidator:
    """Validates fetched market data and computes quality score"""

    def validate(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, float, List[str]]:
        """
        Validate data and compute quality score.

        Returns: (is_valid, quality_score, failed_checks)
        """
        if df is None or df.empty:
            return False, 0.0, ["EMPTY_DATAFRAME"]

        failed_checks = []
        score = 1.0

        # Required columns
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                failed_checks.append(f"MISSING_COLUMN_{col}")
                return False, 0.0, failed_checks

        # 1. Check for duplicate dates
        if df.index.has_duplicates:
            failed_checks.append("DUPLICATE_DATES")
            score *= 0.8

        # 2. Check for zero/negative prices
        if (df['Close'] <= 0).any():
            failed_checks.append("ZERO_NEGATIVE_PRICES")
            score = 0.0

        # 3. Check for zero-volume flatlines
        zero_vol_days = (df['Volume'] == 0).sum()
        if zero_vol_days > len(df) * 0.05:  # More than 5% zero-volume days
            failed_checks.append(f"EXCESS_ZERO_VOLUME_DAYS({zero_vol_days})")
            score *= 0.7

        # 4. Check for flash crashes/spikes (>30% daily move)
        returns = df['Close'].pct_change().dropna()
        if not returns.empty:
            max_return = returns.abs().max()
            if max_return > 0.30:
                failed_checks.append(f"FLASH_SPIKE_DETECTED({max_return:.1%})")
                score *= 0.5

        # 5. Check for flat price sequences (5+ days no change)
        price_changes = df['Close'].diff().abs()
        flat_days = (price_changes == 0).sum()
        if flat_days > len(df) * 0.10:  # More than 10% flat days
            failed_checks.append(f"EXCESS_FLAT_DAYS({flat_days})")
            score *= 0.6

        # 6. Check data freshness (end date should be recent)
        if len(df) > 0:
            latest_date = df.index[-1]
            days_old = (datetime.utcnow() - latest_date).days
            if days_old > 5:
                failed_checks.append(f"STALE_DATA({days_old} days old)")
                score *= 0.9

        # 7. Check for excessive NaN values
        nan_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if nan_pct > 0.05:
            failed_checks.append(f"HIGH_NAN_PERCENTAGE({nan_pct:.1%})")
            score *= 0.6

        # Check minimum data requirement (need ~1000 trading days for 5 years)
        if len(df) < 800:
            failed_checks.append(f"INSUFFICIENT_DATA({len(df)} rows)")
            score *= 0.7

        # Final check: score threshold
        is_valid = score >= MIN_DATA_QUALITY
        if not is_valid and not failed_checks:
            failed_checks.append(f"LOW_QUALITY_SCORE({score:.2f})")

        return is_valid, max(0.0, score), failed_checks


class DataProvider:
    """Base class for data providers"""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self.stats = {'pulls': 0, 'successes': 0, 'failures': 0, 'total_latency': 0}

    def fetch(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data for a symbol. Must be implemented by subclass."""
        raise NotImplementedError

    def fetch_with_fallback(self, symbol: str, start_date: str, end_date: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Try primary, fallback on failure"""
        # Try primary provider
        df = self.fetch(symbol, start_date, end_date)
        if df is not None and not df.empty:
            with self._lock:
                self.stats['pulls'] += 1
                self.stats['successes'] += 1
            return df, self.name

        # Try fallback if available
        fallback = getattr(self, '_fallback', None)
        if fallback:
            df = fallback.fetch(symbol, start_date, end_date)
            if df is not None and not df.empty:
                with self._lock:
                    self.stats['pulls'] += 1
                    self.stats['successes'] += 1
                return df, fallback.name

        with self._lock:
            self.stats['pulls'] += 1
            self.stats['failures'] += 1
        return None, self.name


class YahooDataProvider(DataProvider):
    """Yahoo Finance data provider (free, fallback)"""

    def __init__(self):
        super().__init__("yahoo")
        self._fallback = None

    def fetch(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            if df is not None and not df.empty:
                df.index = df.index.tz_localize(None)  # Remove timezone for consistency
            return df
        except Exception as e:
            logger.debug(f"Yahoo fetch failed for {symbol}: {e}")
            return None


class PolygonDataProvider(DataProvider):
    """Polygon.io data provider (paid, preferred)"""

    def __init__(self, api_key: str = None):
        super().__init__("polygon")
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self._fallback = YahooDataProvider()

    def fetch(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        if not self.api_key:
            return None

        try:
            import polygon
            client = polygon.RestoreClient(self.api_key)
            # This is a placeholder - actual implementation would use polygon REST API
            return None
        except Exception as e:
            logger.debug(f"Polygon fetch failed for {symbol}: {e}")
            return None


class DataIngestionPipeline:
    """
    Main ingestion pipeline for market data.

    Responsibilities:
    1. Read symbols from system universe
    2. Fetch 5 years of OHLCV data for all symbols
    3. Validate data quality
    4. Persist to database
    5. Return structured summary
    """

    def __init__(
        self,
        universe_path: str = "configs/universe.json",
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        max_workers: int = 10
    ):
        self.universe_path = Path(universe_path)
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers

        # Initialize components
        self.validator = DataQualityValidator()
        self.provider = self._init_provider()

        # Database path
        self.db_path = Path("runtime/market_data.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

        # State
        self.run_id = f"ingest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.symbols = self._load_universe()
        self.provider_stats = {}

        logger.info(f"Pipeline initialized for {len(self.symbols)} symbols")

    def _init_provider(self) -> DataProvider:
        """Initialize best available data provider"""
        # Try paid providers first, fallback to free
        polygon_key = os.getenv("POLYGON_API_KEY")
        if polygon_key:
            logger.info("Using Polygon.io (paid provider)")
            return PolygonDataProvider(polygon_key)

        logger.info("Using Yahoo Finance (free provider)")
        return YahooDataProvider()

    def _load_universe(self) -> List[str]:
        """Load symbols from system universe registry"""
        if not self.universe_path.exists():
            raise FileNotFoundError(f"Universe file not found: {self.universe_path}")

        with open(self.universe_path) as f:
            config = json.load(f)

        symbols = config.get('active_tickers', [])
        logger.info(f"Loaded {len(symbols)} symbols from universe")
        return symbols

    def _init_database(self):
        """Initialize SQLite database with required tables"""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # price_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER,
                data_source TEXT,
                raw_hash TEXT,
                ingestion_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')

        # data_quality table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                score REAL NOT NULL,
                failed_checks TEXT,
                provider_used TEXT,
                row_count INTEGER,
                start_date TEXT,
                end_date TEXT,
                ingestion_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # ingestion_audit table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ingestion_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                data_quality_score REAL,
                provider_used TEXT,
                ingestion_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # corporate_actions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corporate_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action_date TEXT NOT NULL,
                action_type TEXT NOT NULL,
                details_json TEXT,
                data_source TEXT,
                ingestion_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def run(self) -> IngestionSummary:
        """
        Run the complete ingestion pipeline.

        Returns:
            IngestionSummary with structured results
        """
        start_time = time.time()
        start_dt = datetime.utcnow()

        logger.info("=" * 80)
        logger.info(f"DATA INGESTION PIPELINE - RUN ID: {self.run_id}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Symbols: {len(self.symbols)}")
        logger.info("=" * 80)

        # Track results
        results: List[IngestionResult] = []
        quality_scores = []
        failed_symbols = []
        invalid_data_symbols = []

        # Process all symbols
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_symbol, symbol): symbol
                for symbol in self.symbols
            }

            completed = 0
            total = len(futures)

            for future in as_completed(futures):
                completed += 1
                symbol = futures[future]

                try:
                    result = future.result()
                    results.append(result)

                    # Track metrics
                    if result.status == "SUCCESS":
                        quality_scores.append(result.data_quality_score)
                    elif result.status == "FAILED":
                        failed_symbols.append({
                            'symbol': symbol,
                            'error': result.error_message or "Unknown"
                        })
                    elif result.status == "INVALID_DATA":
                        invalid_data_symbols.append({
                            'symbol': symbol,
                            'score': result.data_quality_score,
                            'failed_checks': result.failed_checks
                        })

                    # Progress log
                    if completed % 50 == 0 or completed == total:
                        logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

                except Exception as e:
                    logger.error(f"Unexpected error processing {symbol}: {e}")
                    failed_symbols.append({'symbol': symbol, 'error': str(e)})

        # Compute summary
        end_time = time.time()
        end_dt = datetime.utcnow()

        avg_quality = np.mean(quality_scores) if quality_scores else 0.0

        summary = IngestionSummary(
            run_id=self.run_id,
            total_symbols=len(self.symbols),
            successful_fetches=len([r for r in results if r.status == "SUCCESS"]),
            failed_fetches=len(failed_symbols),
            invalid_data_count=len(invalid_data_symbols),
            average_data_quality_score=avg_quality,
            symbols_failed=failed_symbols,
            symbols_invalid_data=invalid_data_symbols,
            provider_health_stats=self._get_provider_stats(),
            duration_seconds=end_time - start_time,
            start_time=start_dt.isoformat(),
            end_time=end_dt.isoformat()
        )

        # Log summary
        self._log_summary(summary)

        return summary

    def _process_symbol(self, symbol: str) -> IngestionResult:
        """Process a single symbol"""
        result = IngestionResult(
            symbol=symbol,
            status="FAILED",
            start_date=self.start_date,
            end_date=self.end_date
        )

        try:
            # Fetch data
            df, provider_used = self.provider.fetch_with_fallback(
                symbol, self.start_date, self.end_date
            )

            result.provider_used = provider_used

            if df is None or df.empty:
                result.error_message = "No data returned from provider"
                self._audit_ingestion(result)
                return result

            # Ensure proper format
            df = self._normalize_dataframe(df, symbol)

            # Validate data
            is_valid, quality_score, failed_checks = self.validator.validate(df, symbol)

            result.data_quality_score = quality_score
            result.row_count = len(df)
            result.failed_checks = failed_checks

            # Persist price data
            self._persist_price_data(df, symbol, provider_used)

            # Persist quality result
            self._persist_quality_result(result)

            # Set status
            if is_valid:
                result.status = "SUCCESS"
            else:
                result.status = "INVALID_DATA"

            # Audit
            self._audit_ingestion(result)

            return result

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Error processing {symbol}: {e}")
            self._audit_ingestion(result)
            return result

    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize dataframe format"""
        df = df.copy()

        # Ensure date index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Ensure required columns exist
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = 0.0

        # Rename to lowercase for consistency
        df.columns = [c.lower() for c in df.columns]

        # Calculate adjusted close (placeholder - in production would apply corporate actions)
        if 'adjusted_close' not in df.columns:
            df['adjusted_close'] = df['close']

        # Sort by date
        df = df.sort_index()

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        return df

    def _persist_price_data(self, df: pd.DataFrame, symbol: str, provider: str):
        """Persist price data to database"""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Generate hash for integrity
        raw_hash = hashlib.sha256(
            json.dumps(df.to_dict(), default=str).encode()
        ).hexdigest()

        timestamp = datetime.utcnow().isoformat()

        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')

            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO price_history
                    (symbol, date, open, high, low, close, adjusted_close, volume, data_source, raw_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, date_str,
                    row.get('open', 0), row.get('high', 0), row.get('low', 0),
                    row.get('close', 0), row.get('adjusted_close', 0),
                    row.get('volume', 0), provider, raw_hash
                ))
            except Exception as e:
                logger.error(f"Failed to persist {symbol} {date_str}: {e}")

        conn.commit()
        conn.close()

    def _persist_quality_result(self, result: IngestionResult):
        """Persist quality validation result"""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO data_quality
            (symbol, score, failed_checks, provider_used, row_count, start_date, end_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.symbol,
            result.data_quality_score,
            json.dumps(result.failed_checks),
            result.provider_used,
            result.row_count,
            result.start_date,
            result.end_date
        ))

        conn.commit()
        conn.close()

    def _audit_ingestion(self, result: IngestionResult):
        """Audit ingestion result"""
        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ingestion_audit
            (run_id, symbol, status, error_message, data_quality_score, provider_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.run_id,
            result.symbol,
            result.status,
            result.error_message,
            result.data_quality_score,
            result.provider_used
        ))

        conn.commit()
        conn.close()

    def _get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get provider statistics"""
        if hasattr(self.provider, 'stats'):
            return {self.provider.name: self.provider.stats}
        return {}

    def _log_summary(self, summary: IngestionSummary):
        """Log and save summary"""
        logger.info("=" * 80)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Run ID: {summary.run_id}")
        logger.info(f"Duration: {summary.duration_seconds:.2f}s")
        logger.info(f"Total Symbols: {summary.total_symbols}")
        logger.info(f"Successful: {summary.successful_fetches}")
        logger.info(f"Failed: {summary.failed_fetches}")
        logger.info(f"Invalid Data: {summary.invalid_data_count}")
        logger.info(f"Avg Quality Score: {summary.average_data_quality_score:.4f}")
        logger.info("=" * 80)

        # Save summary to JSON
        summary_path = Path(f"output/ingestion_summary_{summary.run_id}.json")
        summary_path.parent.mkdir(exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"Summary saved to {summary_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Market Data Ingestion Pipeline')
    parser.add_argument('--universe', type=str, default='configs/universe.json',
                        help='Path to universe JSON file')
    parser.add_argument('--start', type=str, default=DEFAULT_START_DATE,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=DEFAULT_END_DATE,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel workers')

    args = parser.parse_args()

    try:
        pipeline = DataIngestionPipeline(
            universe_path=args.universe,
            start_date=args.start,
            end_date=args.end,
            max_workers=args.workers
        )

        summary = pipeline.run()

        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL OUTPUT")
        print("=" * 80)
        print(json.dumps(summary.to_dict(), indent=2))

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

