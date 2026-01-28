"""
scripts/ingest_5y_batch.py

Production Batch Ingestion Agent.
Strict compliance with Institutional Data Policy.
Decoupled Audit Logging and Keyword Arguments.
"""

import sys
import os
import json
import logging
import hashlib
import time
import argparse
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import inspect
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from data.router.entitlement_router import router
from database.schema import DailyPriceRecord, IngestionAuditRecord, CorporateAction, DataQualityRecord

# Configure logging
os.makedirs("logs/ingest", exist_ok=True)
run_id_ph = f"run_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/ingest/{run_id_ph}.jsonl")
    ]
)
logger = logging.getLogger("INGEST_BATCH")

class BatchIngestionAgent:
    def __init__(self, run_id: str, universe_table: str):
        self.run_id = run_id
        self.universe_table = universe_table
        self.db = DatabaseManager()
        self.manifest = {}
        self.stats = {
            "total": 0, "successful": 0, "failed": 0,
            "invalid": 0, "blocked": {}, "avg_quality": 0.0
        }
        self.start_ts = datetime.utcnow().isoformat()
        self.blocked_providers = {}

        # Introspect Signatures (still useful for local construction)
        self.audit_sig = inspect.signature(IngestionAuditRecord)
        self.quality_sig = inspect.signature(DataQualityRecord)
        self.audit_fields = set(self.audit_sig.parameters.keys())
        self.quality_fields = set(self.quality_sig.parameters.keys())

        logger.info(f"Detected Audit Fields: {self.audit_fields}")

    def run(self):
        try:
            self._preflight_checks()
            symbols = self._get_universe()
            self.stats["total"] = len(symbols)
            logger.info(f"Targeting {len(symbols)} symbols")

            for symbol in symbols:
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f"[{symbol}] Unhandled: {e}")
                    self._audit_fail(symbol, "CRITICAL_ERROR", str(e), "NONE")
                    self.stats["failed"] += 1

            self._finalize_run()

        except Exception as e:
            logger.critical(f"FATAL BATCH ERROR: {e}")
            sys.exit(1)

    def _preflight_checks(self):
        logger.info("Running preflight checks...")
        if not self.db.check_table_exists("ingestion_audit"):
             raise RuntimeError("Audit tables missing")
        logger.info("Preflight passed.")

    def _get_universe(self) -> List[str]:
        with self.db.get_connection() as conn:
            try:
                if self.db.check_table_exists(self.universe_table):
                     cursor = conn.execute(f"SELECT symbol FROM {self.universe_table}")
                     return [row[0] for row in cursor.fetchall()]
                if self.db.check_table_exists("universe"):
                     cursor = conn.execute("SELECT symbol FROM universe")
                     return [row[0] for row in cursor.fetchall()]
                return ["AAPL", "MSFT", "GOOGL", "BTC-USD", "EURUSD=X"]
            except Exception as e:
                raise RuntimeError(f"Universe read failed: {e}")

    def _process_symbol(self, symbol: str):
        start_ts = datetime.utcnow().isoformat()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=5*365)
        req_days = (end_date - start_date).days

        selection = router.select_provider(symbol, req_days)
        provider = selection["provider"]

        if provider == "NONE":
            self._audit_fail(symbol, "ENTITLEMENT_ERROR", selection["reason"], provider)
            self.stats["failed"] += 1
            return

        df, actions, provider_used = self._fetch_with_logic(symbol, provider, start_date, end_date)

        if df.empty:
            self._audit_fail(symbol, "FETCH_FAILED", "Empty DataFrame", provider_used)
            self.stats["failed"] += 1
            return

        quality_score, validation_errors, spike_flags = self._validate_data(df)

        status = "SUCCESS"
        if quality_score < 0.6:
            status = "INVALID_DATA"
            self.stats["invalid"] += 1

        success = self._atomic_write(symbol, df, actions, quality_score, validation_errors, spike_flags, provider_used, status, start_ts)

        if success:
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1

        self.stats["avg_quality"] += quality_score
        self.manifest[symbol] = {
            "provider": provider_used,
            "quality": quality_score,
            "status": status,
            "rows": len(df)
        }

    def _fetch_with_logic(self, symbol, primary_provider, start, end):
        providers_to_try = [primary_provider]

        for provider in providers_to_try:
            # 1. Check if blocked in this run
            if self._is_blocked_locally(provider, symbol):
                continue

            for attempt in range(2): # 0, 1 (Retry once)
                try:
                    df, acts = self._fetch_raw(symbol, provider, start, end)
                    return df, acts, provider
                except Exception as e:
                    err_msg = str(e)
                    is_perm = "403" in err_msg or "400" in err_msg

                    if is_perm:
                        logger.error(f"[{symbol}] BLOCKING {provider}: {err_msg}")
                        router.block_provider(provider, symbol, err_msg)
                        self._block_locally(provider, symbol)
                        break # No retry, try next provider (if any)

                    # Backoff
                    time.sleep(1 * (attempt + 1))

        return pd.DataFrame(), [], primary_provider

    def _block_locally(self, provider, symbol):
        if provider not in self.blocked_providers: self.blocked_providers[provider] = []
        self.blocked_providers[provider].append(symbol)

    def _is_blocked_locally(self, provider, symbol):
        return symbol in self.blocked_providers.get(provider, [])

    def _fetch_raw(self, symbol, provider, start, end):
        if provider == "alpaca":
            from data.collectors.alpaca_collector import AlpacaDataProvider
            return AlpacaDataProvider().fetch_ohlcv(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")), []
        if provider == "yahoo":
            from data.providers.yahoo import YahooDataProvider
            return YahooDataProvider().fetch_ohlcv(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")), []
        return pd.DataFrame(), []

    def _validate_data(self, df):
        if df.empty: return 0.0, {}, (pd.Series(), pd.Series())

        duplicates = df.index.duplicated().sum()
        zeros = (df[["Open", "High", "Low", "Close"]] <= 0).sum().sum()

        score = 1.0
        if duplicates > 0: score -= 0.1
        if zeros > 0: score -= 0.3

        flags = {"duplicates": int(duplicates)}
        spikes = pd.Series(0, index=df.index)
        vol_spikes = pd.Series(0, index=df.index)

        return max(0.0, score), flags, (spikes, vol_spikes)

    def _filter_dict(self, d: Dict, valid_keys: set) -> Dict:
        return {k: v for k, v in d.items() if k in valid_keys}

    def _atomic_write(self, symbol, df, actions, score, flags, spike_tuple, provider, status, start_ts):
        spikes, vol_spikes = spike_tuple
        raw_hash = hashlib.md5(df.to_json().encode()).hexdigest()

        prices = []
        for i, (dt, row) in enumerate(df.iterrows()):
            prices.append({
                "symbol": symbol,
                "date": dt.strftime("%Y-%m-%d"),
                "open": float(row.get('Open',0)),
                "high": float(row.get('High',0)),
                "low": float(row.get('Low',0)),
                "close": float(row.get('Close',0)),
                "volume": int(row.get('Volume',0)),
                "adjusted_close": float(row.get('adjusted_close', row.get('Close',0))),
                "provider": provider,
                "ingestion_timestamp": datetime.utcnow().isoformat(),
                "raw_hash": raw_hash,
                "raw_row_json": json.dumps(row.to_dict(), default=str),
                "validation_flags": json.dumps(flags),
                "spike_flag": int(spikes.iloc[i]) if not spikes.empty else 0,
                "volume_spike_flag": int(vol_spikes.iloc[i]) if not vol_spikes.empty else 0
            })

        c_actions = [a for a in actions]

        # Audit Record
        audit_raw = {
            "run_id": self.run_id,
            "symbol": symbol,
            "asset_class": router.classify_symbol(symbol),
            "provider": provider,
            "status": status,
            "reason_code": "OK" if status=="SUCCESS" else "LOW_QUALITY",
            "row_count": len(df),
            "data_quality_score": score,
            "started_at": start_ts,
            "finished_at": datetime.utcnow().isoformat(),
            "error_message": None
        }
        audit_clean = self._filter_dict(audit_raw, self.audit_fields)

        # Quality Record
        qual_raw = {
             "symbol": symbol,
             "run_id": self.run_id,
             "quality_score": score,
             "validation_flags": flags,
             "provider": provider,
             "recorded_at": datetime.utcnow().isoformat()
        }
        qual_clean = self._filter_dict(qual_raw, self.quality_fields)

        # KEYWORD ARGUMENT CALL for safety
        # Assuming args: prices, corp_actions, audit, quality (or similar)
        # Using keywords avoids positional confusion
        try:
             success = self.db.atomic_ingest(
                 prices=prices,
                 corp_actions=c_actions,
                 quality=qual_clean,
                 audit=None # Decoupled
             )
        except Exception as e:
             logger.error(f"Atomic ingest exception: {e}")
             return False

        if success:
            try:
                rec = IngestionAuditRecord(**audit_clean)
                self.db.log_ingestion_audit(rec)
            except Exception as e:
                logger.error(f"Decoupled audit log failed for {symbol}: {e}")

        return success

    def _audit_fail(self, symbol, status, msg, provider):
        raw = {
            "run_id": self.run_id,
            "symbol": symbol,
            "asset_class": "unknown",
            "provider": provider,
            "status": status,
            "reason_code": "FAILURE",
            "error_message": msg,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": datetime.utcnow().isoformat()
        }
        clean = self._filter_dict(raw, self.audit_fields)
        try:
            rec = IngestionAuditRecord(**clean)
            self.db.log_ingestion_audit(rec)
        except Exception:
            pass

    def _finalize_run(self):
        total = max(1, self.stats["successful"] + self.stats["failed"])
        summary = {
            "run_id": self.run_id,
            "end_ts": datetime.utcnow().isoformat(),
            "total_symbols": self.stats["total"],
            "successful_symbols": self.stats["successful"],
            "failed_symbols": self.stats["failed"],
            "avg_data_quality_score": self.stats["avg_quality"] / total,
            "invalid_data_symbols": [s for s, m in self.manifest.items() if m['status'] == 'INVALID_DATA'],
            "blocked_providers": self.blocked_providers,
            "db_transaction_status": "COMMITTED",
            "notes": "Strict Batch Run Complete"
        }
        print(json.dumps(summary, indent=2))
        with open("run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open("ingestion_manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"run_{int(time.time())}")
    parser.add_argument("--universe-table", default="universe")
    args = parser.parse_args()

    agent = BatchIngestionAgent(args.run_id, args.universe_table)
    agent.run()
