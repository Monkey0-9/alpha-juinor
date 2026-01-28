#!/usr/bin/env python3
"""
5-Year Market Data Backfill Orchestrator.

Fetches, validates, and persists the last 5 years of historical market data
for all configured symbols with full provenance and audit trail.

Usage:
    python tools/backfill_5y.py --start 2021-01-19 --end 2026-01-19 --symbols all
    python tools/backfill_5y.py --status --job_id XYZ
    python tools/backfill_5y.py --resume --job_id XYZ

Authoritative Window: 2021-01-19 through 2026-01-19
"""

import os
import sys
import json
import logging
import hashlib
import uuid
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from threading import Lock

import numpy as np
import pandas as pd

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.schema import (
    DailyPriceRecord,
    IngestionAuditRecord,
    DataQualityRecord,
    ProviderMetricsRecord,
    BackfillFailureRecord,
)
from database.manager import get_db, DatabaseManager
from data.providers.base import DataProvider
from data.providers.yahoo import YahooDataProvider
from data.providers.polygon import PolygonDataProvider
from data.providers.alpha_vantage import AlphaVantageProvider
from data.providers.stooq import StooqProvider
from data_intelligence.provider_bandit import ProviderBandit
from data_intelligence.provider_quota_manager import ProviderQuotaManager
from configs.config_manager import get_config

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/backfill_5y_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MIN_DATA_QUALITY = 0.6
NULL_THRESHOLD = 0.05
DEFAULT_WORKERS = 10
MAX_RETRIES = 5
RETRY_DELAY_BASE = 1  # seconds


@dataclass
class BackfillResult:
    """Result of backfilling a single symbol"""
    symbol: str
    status: str = "pending"  # success, failed, invalid_data
    provider: str = ""
    rows_fetched: int = 0
    quality_score: float = 0.0
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    raw_hash: str = ""


@dataclass
class BackfillJob:
    """Backfill job metadata"""
    job_id: str
    job_type: str = "backfill"
    start_date: str = ""
    end_date: str = ""
    symbols: List[str] = field(default_factory=list)
    status: str = "running"
    started_at: str = ""
    completed_at: str = ""
    duration_ms: float = 0.0
    summary_json: str = ""


class DataValidator:
    """Validates fetched market data and computes quality score"""

    # NYSE trading holidays (approximate)
    TRADING_HOLIDAYS = [
        '2026-01-01', '2026-01-20', '2026-02-17', '2026-04-10', '2026-05-25',
        '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
        '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
        '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
        '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
        '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29',
        '2023-06-19', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
        '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30', '2022-06-20',
        '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26', '2022-01-01',
        '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
        '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24', '2021-12-31',
    ]

    @staticmethod
    def get_trading_days(start_date: str, end_date: str) -> int:
        """Get expected number of trading days between dates"""
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        trading_days = [d for d in all_days if d.strftime('%Y-%m-%d') not in DataValidator.TRADING_HOLIDAYS]
        return len(trading_days)

    def validate(self, df: pd.DataFrame, symbol: str,
                 start_date: str, end_date: str) -> Tuple[bool, float, List[str]]:
        """
        Validate data and compute quality score.

        Returns: (is_valid, quality_score, failed_checks)
        """
        if df is None or df.empty:
            return False, 0.0, ["EMPTY_DATAFRAME"]

        failed_checks = []
        score = 1.0

        # Required columns
        required_cols = ['Close', 'High', 'Low', 'Volume', 'Open']
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
            return False, 0.0, failed_checks

        # 3. Check for zero-volume flatlines
        zero_vol_days = (df['Volume'] == 0).sum()
        if zero_vol_days > len(df) * 0.05:
            failed_checks.append(f"EXCESS_ZERO_VOLUME({zero_vol_days}/{len(df)})")
            score *= 0.7

        # 4. Check for flash crashes/spikes (>30% daily move)
        returns = df['Close'].pct_change().dropna()
        if not returns.empty:
            max_return = returns.abs().max()
            if max_return > 0.30:
                failed_checks.append(f"FLASH_SPIKE_DETECTED({max_return:.1%})")
                score *= 0.5

        # 5. Check data continuity (missing trading days)
        expected_days = self.get_trading_days(start_date, end_date)
        actual_days = len(df)
        missing_pct = max(0, (expected_days - actual_days) / expected_days)
        if missing_pct > 0.10:  # Allow 10% missing
            failed_checks.append(f"MISSING_TRADING_DAYS({actual_days}/{expected_days})")
            score *= 0.8

        # 6. Check for excessive NaN values
        total_cells = len(df) * len(df.columns)
        nan_cells = df.isnull().sum().sum()
        nan_pct = nan_cells / total_cells if total_cells > 0 else 0
        if nan_pct > NULL_THRESHOLD:
            failed_checks.append(f"HIGH_NAN_PERCENTAGE({nan_pct:.1%})")
            score *= 0.6

        # 7. Check price continuity (close_t vs open_{t+1})
        if len(df) > 1:
            close_open_diff = (df['Open'].iloc[1:].values - df['Close'].iloc[:-1].values) / df['Close'].iloc[:-1].values
            large_gaps = np.abs(close_open_diff) > 0.20  # 20% gap
            if large_gaps.any():
                gap_count = large_gaps.sum()
                failed_checks.append(f"PRICE_DISCONTINUITY({gap_count} gaps >20%)")
                score *= 0.7

        # Final check: score threshold
        is_valid = score >= MIN_DATA_QUALITY
        if not is_valid and not failed_checks:
            failed_checks.append(f"LOW_QUALITY_SCORE({score:.2f})")

        return is_valid, max(0.0, score), failed_checks


class ProviderManager:
    """Manages data providers with fallback and MAB selection"""

    def __init__(self, config_path: str = "configs/providers.yaml"):
        self.config = self._load_config(config_path)
        self.quota_manager = ProviderQuotaManager(self.config)
        self.bandit = self._init_bandit()
        self._providers = {}
        self._provider_lock = Lock()

    def _load_config(self, config_path: str) -> Dict:
        """Load provider configuration"""
        from configs.config_manager import get_config
        config = get_config()
        return config.get('providers', {
            'polygon': {'tier': 'primary', 'priority': 1, 'monthly_limit': 50000},
            'alpha_vantage': {'tier': 'primary', 'priority': 2, 'monthly_limit': 25000},
            'stooq': {'tier': 'secondary', 'priority': 3, 'monthly_limit': 100000},
            'yahoo': {'tier': 'fallback', 'priority': 4, 'monthly_limit': 200000},
        })

    def _init_bandit(self) -> ProviderBandit:
        """Initialize provider bandit with configured providers"""
        providers = list(self.config.keys())
        return ProviderBandit(providers, exploration_factor=2.0)

    def get_provider(self, name: str) -> DataProvider:
        """Get or create provider instance"""
        with self._provider_lock:
            if name not in self._providers:
                self._providers[name] = self._create_provider(name)
            return self._providers[name]

    def _create_provider(self, name: str) -> DataProvider:
        """Create provider instance by name"""
        if name == 'polygon':
            return PolygonDataProvider()
        elif name == 'alpha_vantage':
            return AlphaVantageProvider()
        elif name == 'stooq':
            return StooqProvider()
        elif name == 'yahoo':
            return YahooDataProvider()
        else:
            raise ValueError(f"Unknown provider: {name}")

    def select_provider(self, symbol: str) -> Tuple[str, DataProvider]:
        """Select best provider using MAB and quota awareness"""
        # Get provider order by priority
        available = [p for p, cfg in sorted(
            self.config.items(),
            key=lambda x: x[1].get('priority', 4)
        ) if self.quota_manager.check_quota(p)]

        if not available:
            # Fall back to any provider regardless of quota
            available = list(self.config.keys())

        # Use bandit to select
        selected = self.bandit.select_provider(available)
        return selected, self.get_provider(selected)

    def update_bandit(self, provider: str, success: bool,
                      latency_ms: float, quality_score: float):
        """Update bandit statistics after a fetch"""
        self.bandit.update(provider, success, latency_ms, quality_score)

        # Update quota manager
        self.quota_manager.increment_usage(provider)


class BackfillOrchestrator:
    """
    Main orchestrator for 5-year backfill operations.

    Responsibilities:
    1. Manage backfill jobs
    2. Fetch data with provider fallback
    3. Validate and persist data
    4. Emit audit records
    5. Track provider metrics
    """

    def __init__(self, db: DatabaseManager = None, max_workers: int = DEFAULT_WORKERS):
        self.db = db or get_db()
        self.validator = DataValidator()
        self.provider_manager = ProviderManager()
        self.max_workers = max_workers
        self.raw_dir = Path("runtime/raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Thread-safe counters
        self._stats_lock = Lock()
        self.stats = {'success': 0, 'failed': 0, 'invalid': 0, 'total_rows': 0}

    def run_backfill(self, start_date: str, end_date: str,
                     symbols: List[str] = None, job_id: str = None) -> BackfillJob:
        """
        Run full backfill for specified date range and symbols.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols (None = load from universe)
            job_id: Optional job ID (auto-generated if not provided)

        Returns:
            BackfillJob with summary
        """
        job_id = job_id or f"backfill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        logger.info("=" * 80)
        logger.info(f"BACKFILL JOB: {job_id}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 80)

        # Load symbols
        if symbols is None or symbols == ['all']:
            symbols = self._load_universe()

        job = BackfillJob(
            job_id=job_id,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            status='running',
            started_at=datetime.utcnow().isoformat()
        )

        start_time = time.time()
        results: List[BackfillResult] = []

        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._backfill_symbol, symbol, start_date, end_date, job_id): symbol
                for symbol in symbols
            }

            completed = 0
            total = len(futures)

            for future in as_completed(futures):
                completed += 1
                symbol = futures[future]

                try:
                    result = future.result()
                    results.append(result)

                    with self._stats_lock:
                        if result.status == 'success':
                            self.stats['success'] += 1
                            self.stats['total_rows'] += result.rows_fetched
                        elif result.status == 'failed':
                            self.stats['failed'] += 1
                        else:
                            self.stats['invalid'] += 1

                except Exception as e:
                    logger.error(f"Unexpected error for {symbol}: {e}")
                    results.append(BackfillResult(
                        symbol=symbol, status='failed', errors=[str(e)]
                    ))
                    with self._stats_lock:
                        self.stats['failed'] += 1

                # Progress log
                if completed % 50 == 0 or completed == total:
                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

        # Complete job
        end_time = time.time()
        job.duration_ms = (end_time - start_time) * 1000
        job.completed_at = datetime.utcnow().isoformat()
        job.status = 'completed' if self.stats['failed'] == 0 else 'partial'

        # Generate summary
        summary = {
            'job_id': job_id,
            'period': f"{start_date} to {end_date}",
            'symbols_total': len(symbols),
            'symbols_success': self.stats['success'],
            'symbols_failed': self.stats['failed'],
            'symbols_invalid': self.stats['invalid'],
            'total_rows_fetched': self.stats['total_rows'],
            'duration_seconds': round(job.duration_ms / 1000, 2),
            'started_at': job.started_at,
            'completed_at': job.completed_at
        }
        job.summary_json = json.dumps(summary, indent=2)

        # Log completion
        self._log_job_completion(job, summary)

        return job

    def _backfill_symbol(self, symbol: str, start_date: str,
                         end_date: str, job_id: str) -> BackfillResult:
        """Backfill a single symbol with provider fallback"""
        result = BackfillResult(symbol=symbol)
        start_time = time.time()

        try:
            # Select provider using MAB
            provider_name, provider = self.provider_manager.select_provider(symbol)
            result.provider = provider_name

            logger.debug(f"Fetching {symbol} from {provider_name}")

            # Fetch with retry
            df = self._fetch_with_retry(provider, symbol, start_date, end_date)

            if df is None or df.empty:
                result.status = 'failed'
                result.errors.append(f"No data from {provider_name}")
                self._log_failure(job_id, symbol, provider_name, start_date, end_date, result.errors)
                return result

            # Validate data
            is_valid, quality_score, failed_checks = self.validator.validate(
                df, symbol, start_date, end_date
            )
            result.quality_score = quality_score
            result.errors = failed_checks
            result.rows_fetched = len(df)

            # Compute raw hash
            raw_payload = df.to_json(date_format='iso', orient='split')
            result.raw_hash = hashlib.sha256(raw_payload.encode()).hexdigest()

            # Persist raw data
            self._persist_raw_data(symbol, provider_name, start_date, end_date, raw_payload)

            # Upsert to database
            self._persist_to_db(df, symbol, provider_name, result.raw_hash)

            # Log quality
            self._log_quality(symbol, start_date, end_date, quality_score,
                             failed_checks, provider_name, len(df))

            # Update provider metrics
            duration_ms = (time.time() - start_time) * 1000
            self.provider_manager.update_bandit(provider_name, is_valid,
                                                 duration_ms, quality_score)

            # Set status
            result.status = 'success' if is_valid else 'invalid_data'

            # Log audit record
            self._log_audit(job_id, result, start_date, end_date)

            return result

        except Exception as e:
            result.status = 'failed'
            result.errors.append(str(e))
            logger.error(f"Error backfilling {symbol}: {e}")
            self._log_failure(job_id, symbol, result.provider or 'unknown',
                             start_date, end_date, [str(e)])
            return result

    def _fetch_with_retry(self, provider: DataProvider, symbol: str,
                          start_date: str, end_date: str,
                          max_retries: int = MAX_RETRIES) -> Optional[pd.DataFrame]:
        """Fetch data with exponential backoff retry"""
        last_error = None

        for attempt in range(max_retries):
            try:
                df = provider.fetch_ohlcv(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    return df

                last_error = f"Empty response from {provider.__class__.__name__}"

            except Exception as e:
                last_error = str(e)
                logger.debug(f"Attempt {attempt + 1} failed for {symbol}: {e}")

            # Exponential backoff
            if attempt < max_retries - 1:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                time.sleep(delay)

        logger.warning(f"All retries failed for {symbol}: {last_error}")
        return None

    def _persist_raw_data(self, symbol: str, provider: str,
                          start_date: str, end_date: str, raw_payload: str):
        """Persist raw response to disk"""
        try:
            # Path: runtime/raw/YYYY-MM-DD/symbol/provider/response.json.gz
            date_dir = self.raw_dir / start_date
            date_dir.mkdir(parents=True, exist_ok=True)

            provider_dir = date_dir / symbol / provider
            provider_dir.mkdir(parents=True, exist_ok=True)

            file_path = provider_dir / "response.json"
            with open(file_path, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'provider': provider,
                    'start_date': start_date,
                    'end_date': end_date,
                    'fetched_at': datetime.utcnow().isoformat(),
                    'data': json.loads(raw_payload)
                }, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to persist raw data for {symbol}: {e}")

    def _persist_to_db(self, df: pd.DataFrame, symbol: str,
                       provider: str, raw_hash: str):
        """Persist validated data to database"""
        pulled_at = datetime.utcnow().isoformat()
        records = []

        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            records.append(DailyPriceRecord(
                symbol=symbol,
                date=date_str,
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

        self.db.upsert_daily_prices_batch(records)

    def _log_quality(self, symbol: str, start_date: str, end_date: str,
                     quality_score: float, issues: List[str],
                     provider: str, row_count: int):
        """Log data quality assessment"""
        record = DataQualityRecord(
            symbol=symbol,
            date=f"{start_date}_{end_date}",
            quality_score=quality_score,
            issues=issues if issues else None,
            provider=provider,
            row_count=row_count
        )
        self.db.log_data_quality(record)

    def _log_audit(self, job_id: str, result: BackfillResult,
                   start_date: str, end_date: str):
        """Log ingestion audit record"""
        record = IngestionAuditRecord(
            job_id=job_id,
            symbol=result.symbol,
            provider=result.provider,
            start_date=start_date,
            end_date=end_date,
            rows_expected=self.validator.get_trading_days(start_date, end_date),
            rows_fetched=result.rows_fetched,
            raw_hash=result.raw_hash,
            duration_ms=result.duration_ms,
            status=result.status,
            errors=result.errors if result.errors else None
        )
        self.db.log_ingestion_audit(record)

    def _log_failure(self, job_id: str, symbol: str, provider: str,
                     start_date: str, end_date: str, errors: List[str]):
        """Log persistent backfill failure"""
        record = BackfillFailureRecord(
            job_id=job_id,
            symbol=symbol,
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            error_message='; '.join(errors),
            status='failed'
        )
        self.db.log_backfill_failure(record)

    def _log_job_completion(self, job: BackfillJob, summary: Dict):
        """Log job completion to database"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ingestion_jobs
            (job_id, job_type, start_date, end_date, symbols_count,
             status, started_at, completed_at, duration_ms, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.job_id, job.job_type, job.start_date, job.end_date,
            len(job.symbols), job.status, job.started_at, job.completed_at,
            job.duration_ms, job.summary_json
        ))
        conn.commit()

        logger.info(f"Job {job.job_id} completed: {summary}")

    def _load_universe(self) -> List[str]:
        """Load symbols from universe config"""
        with open("configs/universe.json") as f:
            config = json.load(f)
        return config.get('active_tickers', [])

    def get_job_status(self, job_id: str) -> Dict:
        """Get status of a backfill job"""
        conn = self.db.get_connection()
        cursor = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE job_id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return {}

    def get_failed_symbols(self, job_id: str) -> List[Dict]:
        """Get symbols that failed in a job"""
        return self.db.get_backfill_failures(job_id)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='5-Year Market Data Backfill')
    parser.add_argument('--start', type=str, default='2021-01-19',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2026-01-19',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, default='all',
                        help='Symbols to backfill (comma-separated or "all")')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='Number of parallel workers')
    parser.add_argument('--job_id', type=str, default=None,
                        help='Job ID for resumption')
    parser.add_argument('--status', action='store_true',
                        help='Check job status')

    args = parser.parse_args()

    # Handle status check
    if args.status:
        if not args.job_id:
            parser.error("--status requires --job_id")

        orchestrator = BackfillOrchestrator()
        status = orchestrator.get_job_status(args.job_id)
        print(json.dumps(status, indent=2))
        return 0

    # Parse symbols
    if args.symbols == 'all':
        symbols = ['all']
    else:
        symbols = [s.strip() for s in args.symbols.split(',')]

    # Run backfill
    orchestrator = BackfillOrchestrator(max_workers=args.workers)
    job = orchestrator.run_backfill(
        start_date=args.start,
        end_date=args.end,
        symbols=symbols if symbols != ['all'] else None,
        job_id=args.job_id
    )

    # Print summary
    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE")
    print("=" * 80)
    print(json.dumps(json.loads(job.summary_json), indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

