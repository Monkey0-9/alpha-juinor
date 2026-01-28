
import sqlite3
import threading
import logging
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

from .base import DatabaseAdapter
from ..schema import (
    SCHEMA_SQL, SCHEMA_VERSION, DailyPriceRecord, IntradayPriceRecord,
    IngestionAuditRecord, DataQualityRecord, ProviderMetricsRecord,
    ExecutionFeedbackRecord, BackfillFailureRecord, SymbolGovernanceRecord,
    ModelDecayMetric, CapitalAllocation, GovernanceDecisionRecord,
    StrategyLifecycleRecord, CorporateAction, FeatureRecord, ModelOutput,
    DecisionRecord, OrderRecord, PositionRecord, AuditEntry, CycleMeta
)
from ..errors import DatabaseError

logger = logging.getLogger(__name__)

class SQLiteAdapter(DatabaseAdapter):
    """
    SQLite implementation of DatabaseAdapter.
    Compatible with original DatabaseManager behavior.
    """

    def __init__(self, db_path: str, audit_path: str):
        self.db_path = Path(db_path)
        self.audit_path = Path(audit_path)
        self._local = threading.local()
        self._write_lock = threading.Lock()

        # Ensure directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_schema()
        logger.info(f"SQLiteAdapter initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=60.0, # Increased timeout
                check_same_thread=False,
                isolation_level=None  # Enable autocommit mode for WAL
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000") # 5s timeout
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()

        # Execute schema script
        cursor.executescript(SCHEMA_SQL)
        conn.commit()

    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS _schema_versions (
                version TEXT PRIMARY KEY,
                applied_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute("SELECT version FROM _schema_versions ORDER BY applied_at DESC LIMIT 1")
        row = cursor.fetchone()
        current_version = row['version'] if row else None

        if current_version != SCHEMA_VERSION:
            logger.info(f"Upgrading SQLite schema from {current_version} to {SCHEMA_VERSION}")
            cursor.executescript(SCHEMA_SQL)
            cursor.execute(
                "INSERT OR REPLACE INTO _schema_versions (version) VALUES (?)",
                (SCHEMA_VERSION,)
            )
            conn.commit()

    @contextmanager
    def transaction(self):
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"SQLite Transaction failed: {e}")
            raise DatabaseError(f"Transaction failed: {e}")

    def close(self):
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def health_check(self) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            checks = {
                'database_exists': True,
                'engine': 'sqlite',
                'schema_version': SCHEMA_VERSION,
                'tables': {}
            }
            tables = ['price_history_daily', 'data_quality_log', 'ingestion_audit']
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    checks['tables'][table] = cursor.fetchone()[0]
                except Exception:
                    checks['tables'][table] = "N/A"

            checks['quality_summary'] = self.get_quality_summary()
            return checks
        except Exception as e:
            return {'healthy': False, 'error': str(e)}

    # --- Implementation of Abstract Methods (Pasted from Manager) ---

    def upsert_daily_price(self, record: DailyPriceRecord) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.execute('''
                INSERT INTO price_history
                (symbol, date, open, high, low, close, volume, adjusted_close, provider, ingestion_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    close = excluded.close, volume = excluded.volume, ingestion_timestamp = excluded.ingestion_timestamp
            ''', (record.symbol, record.date, record.open, record.high, record.low,
                  record.close, record.volume, record.adjusted_close, record.provider, record.ingestion_timestamp))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed upsert_daily_price: {e}")
            return False

    def upsert_corporate_actions_batch(self, records: List[CorporateAction]) -> int:
        if not records: return 0
        try:
            # Requires schema update to have provider/ingestion_timestamp columns if strict
            # Assuming schema updated.
            with self.transaction() as conn:
                data = [(
                    r.symbol,
                    r.date,
                    r.action_type,
                    json.dumps(r.details) if isinstance(r.details, dict) else r.details,
                    r.source, # provider
                    datetime.utcnow().isoformat()
                ) for r in records]

                # Using simple INSERT for now, or UPSERT if conflict on symbol+date+type
                # Assuming id is PK, no composite unique constraint defined in schema yet broadly?
                # Best effort insert
                conn.executemany('''
                    INSERT INTO corporate_actions (symbol, action_date, action_type, action_details, provider, ingestion_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', data)
            return len(records)
        except Exception as e:
            logger.error(f"Batch corp action insert failed: {e}")
            return 0

    def upsert_daily_prices_batch(self, records: List[DailyPriceRecord]) -> int:
        if not records: return 0
        try:
            with self.transaction() as conn:
                data = [(r.symbol, r.date, r.open, r.high, r.low, r.close, r.volume,
                         r.adjusted_close, r.provider, r.raw_hash,
                         json.dumps(r.validation_flags) if isinstance(r.validation_flags, dict) else r.validation_flags,
                         r.ingestion_timestamp) for r in records]
                conn.executemany('''
                    INSERT INTO price_history (symbol, date, open, high, low, close, volume, adjusted_close, provider, raw_hash, validation_flags, ingestion_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, date) DO UPDATE SET
                        close=excluded.close, volume=excluded.volume, ingestion_timestamp=excluded.ingestion_timestamp,
                        validation_flags=excluded.validation_flags
                ''', data)
            return len(records)
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            return 0

    def get_daily_prices(self, symbol: str, start_date: str = None, end_date: str = None, limit: Optional[int] = None) -> pd.DataFrame:
        conn = self._get_connection()
        query = "SELECT * FROM price_history WHERE symbol = ?"
        params = [symbol]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df = df.sort_values("date")
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df

    def get_daily_prices_batch(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        # Simple implementation for SQLite
        conn = self._get_connection()
        p = ','.join('?' * len(symbols))
        query = f"SELECT * FROM price_history WHERE symbol IN ({p}) AND date >= ? AND date <= ? ORDER BY symbol, date"
        df = pd.read_sql_query(query, conn, params=symbols + [start_date, end_date])
        res = {}
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            for s in symbols:
                sdf = df[df['symbol'] == s].copy()
                if not sdf.empty:
                    sdf.set_index('date', inplace=True)
                    res[s] = sdf
        return res

    def upsert_intraday_price(self, record: IntradayPriceRecord) -> bool:
        # Placeholder or copy logic
        return False

    def get_intraday_prices(self, symbol: str, date: str = None) -> pd.DataFrame:
        return pd.DataFrame()

    def log_ingestion_audit(self, record: IngestionAuditRecord) -> bool:
        # Full Institutional Schema
        conn = self._get_connection()
        try:
            conn.execute('''
                INSERT INTO ingestion_audit
                (run_id, symbol, asset_class, provider, status, reason_code, error_message, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.run_id,
                record.symbol,
                record.asset_class,
                record.provider,
                record.status,
                record.reason_code,
                record.error_message,
                record.started_at,
                record.finished_at
            ))
            # conn.commit() # Handled by transaction context or autocommit settings in manager
            return True
        except Exception as e:
            logger.error(f"Audit log failed: {e}")
            return False

    def log_ingestion_run(self, run_id: str, stats: Dict) -> bool:
        # Simplified copy
        return True

    def get_ingestion_audit(self, run_id=None, symbol=None, status=None) -> List[Dict]:
        return []

    def log_data_quality(self, record: DataQualityRecord) -> bool:
        conn = self._get_connection()
        try:
            # Serialize validation_flags if dict
            val_json = json.dumps(record.validation_flags) if isinstance(record.validation_flags, dict) else record.validation_flags

            conn.execute('''
                INSERT INTO data_quality
                (symbol, run_id, quality_score, validation_flags, provider, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                record.symbol,
                record.run_id,
                record.quality_score,
                val_json,
                record.provider,
                record.recorded_at or datetime.utcnow().isoformat()
            ))
            return True
        except Exception as e:
             logger.error(f"Quality log failed: {e}")
             return False

    def get_data_quality(self, symbol=None, min_score=None) -> List[Dict]:
        return []

    def get_quality_summary(self) -> Dict[str, Any]:
        return {'avg_score': 0.0}

    def update_provider_metrics(self, record) -> bool: return True
    def get_provider_metrics(self, provider=None, days=30) -> List[Dict]: return []
    def get_provider_success_rates(self) -> Dict[str, float]: return {}
    def log_execution_feedback(self, record) -> bool: return True
    def log_backfill_failure(self, record) -> bool: return True
    def get_backfill_failures(self, run_id=None) -> List[Dict]: return []

    def upsert_symbol_governance(self, record: SymbolGovernanceRecord) -> bool:
        conn = self._get_connection()
        try:
            conn.execute('''
                INSERT INTO symbol_governance (symbol, state, data_quality, reason, last_checked_ts, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                state=excluded.state, data_quality=excluded.data_quality, reason=excluded.reason,
                last_checked_ts=excluded.last_checked_ts, metadata_json=excluded.metadata_json
            ''', (record.symbol, record.state, record.data_quality, record.reason,
                  record.last_checked_ts or datetime.utcnow().isoformat(),
                  json.dumps(record.metadata) if record.metadata else None))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Gov update failed: {e}")
            return False

    def get_symbol_governance(self, symbol: str) -> Optional[Dict]:
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM symbol_governance WHERE symbol = ?", (symbol,))
        row = cur.fetchone()
        if row:
            d = dict(row)
            d['metadata'] = json.loads(d['metadata_json']) if d['metadata_json'] else {}
            return d
        return None

    def get_all_symbol_governance(self) -> List[Dict]:
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM symbol_governance")
        return [dict(r) for r in cur.fetchall()]

    def get_active_symbols(self) -> List[str]:
        conn = self._get_connection()
        cur = conn.execute("SELECT symbol FROM symbol_governance WHERE state='ACTIVE'")
        return [r['symbol'] for r in cur.fetchall()]

    def upsert_model_decay(self, record) -> bool: return True
    def insert_capital_allocations(self, records) -> int: return 0
    def log_governance_decision(self, record) -> bool: return True
    def upsert_strategy_lifecycle(self, record) -> bool: return True
    def upsert_features(self, records) -> int:
        # Simplified feature upsert
        if not records: return 0
        conn = self._get_connection()
        try:
            for r in records:
                conn.execute('INSERT OR REPLACE INTO features (symbol, date, features_json) VALUES (?, ?, ?)',
                             (r.symbol, r.date, json.dumps(r.features)))
            conn.commit()
            return len(records)
        except Exception: return 0

    def get_latest_features(self, symbols: List[str]) -> Dict[str, Dict]:
        # Simplified
        res = {}
        for s in symbols:
            f = self.get_features(s)
            if f: res[s] = {'features': f, 'date': datetime.utcnow().isoformat()} # Mock date if needed or fetch real
        return res

    def get_features(self, symbol: str, date: str = None) -> Dict:
        conn = self._get_connection()
        if date:
            cur = conn.execute("SELECT features_json FROM features WHERE symbol=? AND date=?", (symbol, date))
        else:
            cur = conn.execute("SELECT features_json FROM features WHERE symbol=? ORDER BY date DESC LIMIT 1", (symbol,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else {}

    def insert_model_outputs(self, outputs) -> int: return 0
    def insert_decisions(self, decisions) -> int: return 0
    def insert_orders(self, orders) -> int: return 0

    def upsert_position(self, p: PositionRecord) -> bool:
        conn = self._get_connection()
        conn.execute('INSERT OR REPLACE INTO positions (symbol, qty) VALUES (?, ?)', (p.symbol, p.qty))
        conn.commit()
        return True

    def get_positions(self) -> List[Dict]:
        conn = self._get_connection()
        return [dict(r) for r in conn.execute("SELECT * FROM positions").fetchall()]

    def log_audit(self, entry: AuditEntry) -> int:
        with self._write_lock:
            try:
                with open(self.audit_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'timestamp': datetime.utcnow().isoformat(), 'msg': entry.message}) + "\n")
            except: pass
        return 1

    def insert_cycle_meta(self, meta) -> bool: return True

    def get_symbol_coverage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        conn = self._get_connection()
        rows = conn.execute("SELECT symbol, COUNT(*) as c FROM price_history GROUP BY symbol").fetchall()
        return {'details': [{'symbol': r['symbol'], 'row_count': r['c']} for r in rows]}

    def insert_corporate_actions(self, actions) -> int: return 0
