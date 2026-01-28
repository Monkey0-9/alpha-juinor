"""
Institutional Market Data Ingestion Agent.
Handles 5-year historical data ingestion with strict governance and quality rules.
"""

import gzip
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data.collectors.data_router import DataRouter
from database.manager import DatabaseManager
from database.schema import DailyPriceRecord, DataQualityRecord, IngestionAuditRecord

logger = logging.getLogger(__name__)


class TokenBucket:
    """Simple Token Bucket for Rate Limiting."""

    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: int = 1):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: int = 1):
        while not self.consume(tokens):
            time.sleep(0.1)


class InstitutionalIngestionAgent:
    """
    Mandated Institutional Ingestion Agent.
    Strictly adheres to failure handling, archiving, and quality scoring rules.
    """

    def __init__(self, tickers: List[str] = None, run_id: str = None):
        # run_id = run_{UTC_ISO}_{sha1(shortlist)}
        if run_id:
            self.run_id = run_id
        else:
            shortlist = ",".join(sorted(tickers[:5])) if tickers else "no_tickers"
            short_sha = hashlib.sha1(shortlist.encode()).hexdigest()[:8]
            self.run_id = (
                f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{short_sha}"
            )

        self.db = DatabaseManager()
        self.router = DataRouter(max_workers=32, enable_cache=False)

        # Throttling Buckets
        self.throttlers = {
            "polygon": TokenBucket(rate=20, capacity=20),
            "yahoo": TokenBucket(rate=5, capacity=5),
            "alpaca": TokenBucket(rate=10, capacity=10),
        }

        self.stats = {
            "total_symbols": len(tickers) if tickers else 0,
            "processed": 0,
            "successful": 0,
            "rejected": 0,
            "failed": 0,
            "quality_scores": [],
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
        }
        self._stats_lock = threading.Lock()

    def _archive_raw_response(self, symbol: str, provider: str, data: Any):
        """Archive raw provider response to GZIP JSON."""
        archive_dir = f"runtime/raw/{self.run_id}"
        os.makedirs(archive_dir, exist_ok=True)
        filename = f"{symbol}_{provider}_{datetime.utcnow().strftime('%Y%m%d')}.json.gz"
        filepath = os.path.join(archive_dir, filename)

        try:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to archive raw response for {symbol}: {e}")

    def calculate_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Mandated Data Quality Scoring Rule."""
        if df.empty:
            return {"score": 0.0, "flags": {"empty": True}}

        # 1. Missing Dates (Expected trading days in 5 years ~ 1260)
        expected_days = 252 * 5
        missing_dates_pct = max(0.0, (expected_days - len(df)) / expected_days)

        # 2. Duplicate Pct
        duplicate_pct = df.index.duplicated().sum() / len(df) if len(df) > 0 else 0.0

        # 3. Zero/Negative Prices
        zero_negative_flag = (
            1.0 if (df[["Open", "High", "Low", "Close"]] <= 0).any().any() else 0.0
        )

        # 4. Extreme Spike Flag (>6σ volume spike)
        volume = df["Volume"]
        if len(volume) > 1:
            vol_mean = volume.mean()
            vol_std = volume.std()
            extreme_spike_flag = 1.0 if (volume > vol_mean + 6 * vol_std).any() else 0.0
        else:
            extreme_spike_flag = 0.0

        # Weighted Score (Strict Institutional Formula - Tuned)
        # score = 1.0 - (missing_dates_pct * 0.3 + duplicate_pct * 0.2 + zero_negative_flag * 0.2 + extreme_spike_flag * 0.05)
        penalty = (
            missing_dates_pct * 0.3
            + duplicate_pct * 0.2
            + zero_negative_flag * 0.2
            + extreme_spike_flag * 0.05
        )
        score = 1.0 - penalty
        score = max(0.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "flags": {
                "missing_dates_pct": round(missing_dates_pct, 4),
                "duplicate_pct": round(duplicate_pct, 4),
                "zero_negative_flag": bool(zero_negative_flag),
                "extreme_spike_flag": bool(extreme_spike_flag),
            },
        }

    def ingest_symbol(self, symbol: str):
        """Transactional Ingestion Logic for a single symbol."""
        started_at = datetime.utcnow().isoformat()
        asset_class = self.router._classify_ticker(symbol)
        history_days = int(
            6.0 * 365
        )  # Approx 2190 days to ensure FULL 5 years (1260 trading days) of history

        provider = self.router.select_provider(symbol, history_days=history_days)

        if provider == "NO_VALID_PROVIDER":
            self._log_audit(
                symbol,
                asset_class,
                "NONE",
                "REJECTED",
                "NO_VALID_PROVIDER",
                "No provider entitled/available for request",
                started_at,
            )
            with self._stats_lock:
                self.stats["rejected"] += 1
                self.stats["processed"] += 1
            return

        # Throttling
        if provider in self.throttlers:
            self.throttlers[provider].wait_and_consume()

        try:
            # Fetch
            # Calculate start date for institutional backfill
            start_date = (datetime.utcnow() - timedelta(days=history_days)).strftime(
                "%Y-%m-%d"
            )
            df = self.router.get_price_history(
                symbol, start_date=start_date, allow_long_history=True
            )

            # Archive (Mocking raw data as dict for now, usually get_price_history should return raw or we fetch separately)
            # In this architecture, DataRouter returns DataFrame. To adhere perfectly to 'raw_ingest_archive',
            # we'd need providers to return raw JSON too. For now, archive the DF as JSON.
            self._archive_raw_response(symbol, provider, df.to_dict(orient="records"))

            if df.empty:
                raise ValueError("EMPTY_DATASET")

            # Validate & Score
            quality = self.calculate_quality_score(df)
            status = "SUCCESS" if quality["score"] >= 0.6 else "REJECTED"
            reason = None if status == "SUCCESS" else "INVALID_DATA"

            # Transactional Persist (Week 2 Hardening)
            price_records_dicts = []
            for dt, row in df.iterrows():
                price_records_dicts.append({
                    "symbol": symbol,
                    "date": dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else str(dt),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "adjusted_close": float(row.get("adjusted_close", row["Close"])),
                    "provider": provider,
                    "ingestion_timestamp": datetime.utcnow().isoformat(),
                    "raw_hash": self.run_id
                })

            quality_record_dict = {
                "symbol": symbol,
                "run_id": self.run_id,
                "quality_score": quality["score"],
                "validation_flags": quality["flags"],
                "provider": provider
            }

            audit_record_dict = {
                "run_id": self.run_id,
                "symbol": symbol,
                "asset_class": asset_class,
                "provider": provider,
                "status": status,
                "reason_code": reason,
                "error_message": None,
                "started_at": started_at,
                "finished_at": datetime.utcnow().isoformat()
            }

            # Atomic Write: Prices + Quality + Audit
            success = self.db.atomic_ingest(
                prices=price_records_dicts,
                quality=quality_record_dict,
                audit=audit_record_dict
            )

            if success:
                with self._stats_lock:
                    if status == "SUCCESS":
                        self.stats["successful"] += 1
                    else:
                        self.stats["rejected"] += 1
                    self.stats["processed"] += 1
                    self.stats["quality_scores"].append(quality["score"])
            else:
                 # If atomic ingest failed, log fallback audit (best effort)
                 self._log_audit(symbol, asset_class, provider, "FAILED", "DB_WRITE_ERROR", "Atomic ingest failed", started_at)
                 with self._stats_lock:
                     self.stats["failed"] += 1
                     self.stats["processed"] += 1

        except Exception as e:
            error_msg = str(e)
            # Fail-Fast on 403/400 (Handled in Router but catching here for audit)
            if "403" in error_msg or "400" in error_msg:
                self._log_audit(
                    symbol,
                    asset_class,
                    provider,
                    "REJECTED",
                    "ENTITLEMENT_FAILURE",
                    error_msg,
                    started_at,
                )
                with self._stats_lock:
                    self.stats["rejected"] += 1
            else:
                self._log_audit(
                    symbol,
                    asset_class,
                    provider,
                    "FAILED",
                    "PROVIDER_ERROR",
                    error_msg,
                    started_at,
                )
                with self._stats_lock:
                    self.stats["failed"] += 1

            with self._stats_lock:
                self.stats["processed"] += 1

    def _log_audit(
        self, symbol, asset_class, provider, status, reason, error, started_at
    ):
        """Helper to log ingestion audit."""
        self.db.log_ingestion_audit(
            IngestionAuditRecord(
                run_id=self.run_id,
                symbol=symbol,
                asset_class=asset_class,
                provider=provider,
                status=status,
                reason_code=reason,
                error_message=error,
                started_at=started_at,
                finished_at=datetime.utcnow().isoformat(),
            )
        )

    def run_full_universe(self, tickers: List[str]):
        """Parallel Batch Ingestion."""
        print(f"\n[INGESTION_START] Run ID: {self.run_id}")
        print(f"Target Universe: {len(tickers)} symbols\n")

        with ThreadPoolExecutor(max_workers=32) as executor:
            executor.map(self.ingest_symbol, tickers)

        self.stats["end_time"] = datetime.utcnow().isoformat()
        self.finalize_run()

    def finalize_run(self):
        """Mandated JSON Summary and Dashboard Output."""
        avg_q = (
            np.mean(self.stats["quality_scores"])
            if self.stats["quality_scores"]
            else 0.0
        )

        summary = {
            "run_id": self.run_id,
            "total_symbols": self.stats["total_symbols"],
            "processed": self.stats["processed"],
            "successful": self.stats["successful"],
            "rejected": self.stats["rejected"],
            "failed": self.stats["failed"],
            "avg_data_quality": round(float(avg_q), 4),
            "start_time": self.stats["start_time"],
            "end_time": self.stats["end_time"],
        }

        # Persist to DB
        self.db.log_ingestion_run(self.run_id, summary)

        # Mandated Terminal Dashboard
        print("\n" + "=" * 60)
        print("=== DATA HEALTH SUMMARY ===")
        print(f"RUN ID:    {self.run_id}")
        print(f"SUCCESS:   {summary['successful']} / {summary['total_symbols']}")
        print(f"REJECTED:  {summary['rejected']} (Governance/Quality)")
        print(f"FAILED:    {summary['failed']} (Technical Errors)")
        print(f"AVG QUALITY: {summary['avg_data_quality'] * 100:.1f}%")
        print("=" * 60 + "\n")

        # Mandatory Alert Check (>5% Invalid Data)
        if (
            summary["total_symbols"] > 0
            and (summary["rejected"] / summary["total_symbols"]) > 0.05
        ):
            print(f"!! ALERT !! >5% of symbols rejected. Review 'data_quality' table.")

        # WEEK 3: AUTOMATED KILL SWITCH CHECK
        from governance.kill_switch import AutoKillSwitch
        ks = AutoKillSwitch(self.db)
        if not ks.check_slas(summary):
             print("!!! SYSTEM HALTED BY SLA MONITOR !!!")
             # We rely on the kill file existence to stop downstream components (like main.py)

        return summary


if __name__ == "__main__":
    from data.universe_manager import UnifiedUniverseManager

    mgr = UnifiedUniverseManager()
    tickers = mgr.get_active_universe()
    agent = InstitutionalIngestionAgent(tickers)
    agent.run_full_universe(tickers)
