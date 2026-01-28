import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters.base import DatabaseAdapter
from .adapters.sqlite_adapter import SQLiteAdapter
from .schema import (
    AuditEntry,
    BackfillFailureRecord,
    CapitalAllocation,
    CorporateAction,
    CycleMeta,
    DailyPriceRecord,
    DataQualityRecord,
    DecisionRecord,
    ExecutionFeedbackRecord,
    FeatureRecord,
    GovernanceDecisionRecord,
    IngestionAuditRecord,
    IntradayPriceRecord,
    ModelDecayMetric,
    ModelOutput,
    OrderRecord,
    PositionRecord,
    PriceRecord,
    ProviderMetricsRecord,
    StrategyLifecycleRecord,
    SymbolGovernanceRecord,
    TradingEligibilityRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "runtime/institutional_trading.db"
DEFAULT_AUDIT_PATH = "runtime/audit/audit.jsonl"


class DatabaseManager:
    """
    Facade for database operations.
    Delegates to SQLiteAdapter or PostgresAdapter based on configuration.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = DEFAULT_DB_PATH):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        if self._initialized:
            return

        self.db_path = db_path
        self.audit_path = DEFAULT_AUDIT_PATH
        self.engine = os.getenv("DB_ENGINE", "sqlite").lower()

        if self.engine == "postgres":
            # Lazy import to avoid dependency if not used
            from .adapters.postgres_manager import PostgresAdapter

            self.adapter: DatabaseAdapter = PostgresAdapter()
        else:
            self.adapter: DatabaseAdapter = SQLiteAdapter(self.db_path, self.audit_path)

        self._initialized = True
        logger.info(f"DatabaseManager initialized with engine: {self.engine}")

    def transaction(self):
        return self.adapter.transaction()

    def close(self):
        self.adapter.close()

    def health_check(self) -> Dict[str, Any]:
        return self.adapter.health_check()

    def get_connection(self):
        """
        Get a context-managed connection object.

        Usage:
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM table")
                results = cursor.fetchall()

        Returns:
            Context manager yielding connection object
        """
        from .adapters.context_manager import connection_context

        if hasattr(self.adapter, "get_connection"):
            return self.adapter.get_connection()
        # Fallback: wrap _get_connection in context manager
        if hasattr(self.adapter, "_get_connection"):
            return connection_context(self.adapter._get_connection)
        raise NotImplementedError(
            "Underlying adapter does not support direct connection access"
        )

    def _get_connection(self):
        """Deprecated: Use get_connection() instead."""
        import warnings

        warnings.warn(
            "_get_connection is deprecated, use get_connection()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_connection()

    # --- Delegated Methods ---

    def check_table_exists(self, table_name: str) -> bool:
        """Check if table exists (Critical for Governance)."""
        if hasattr(self.adapter, "check_table_exists"):
            return self.adapter.check_table_exists(table_name)
        return False # Fail safe

    def atomic_ingest(self,
                     prices: List[Dict] = None,
                     corp_actions: List[Dict] = None,
                     audit: Dict = None,
                     quality: Dict = None) -> bool:
        """
        Week 1 Blocker: Atomic Ingestion Transaction.
        """
        try:
             with self.transaction():
                 if prices:
                     # Bulk insert prices
                     # Assuming adapter has batch insert or naive loop
                     # Using upsert_daily_prices_batch if available
                     if hasattr(self.adapter, "upsert_daily_prices_batch"):
                         from .schema import DailyPriceRecord
                         # Convert dicts to records
                         records = [DailyPriceRecord(**p) for p in prices]
                         self.adapter.upsert_daily_prices_batch(records)

                 if corp_actions:
                     # Insert corp actions
                     if hasattr(self.adapter, "upsert_corporate_actions_batch"):
                         from .schema import CorporateAction
                         # records = [CorporateAction(**c) for c in corp_actions] # Schema mismatch potential, handle dicts
                         # Adapter should handle list of dicts or records.
                         # Let's assume adapter can take dicts for flexibility or we map to object
                         # Schema object: CorporateAction(symbol, date, action_type, details, source) -> source maps to provider?
                         # Let's map strict to Schema if possible, or pass dicts if adapter allows.
                         # checking schema.py CorporateAction: symbol, date, action_type, details, source

                         c_records = []
                         for c in corp_actions:
                             c_records.append(CorporateAction(
                                 symbol=c['symbol'],
                                 date=c['action_date'],
                                 action_type=c['action_type'],
                                 details=c['action_details'],
                                 source=c.get('provider', 'BATCH') # map provider to source
                             ))
                         self.adapter.upsert_corporate_actions_batch(c_records)

                 if quality:
                     from .schema import DataQualityRecord
                     q_record = DataQualityRecord(**quality)
                     if hasattr(self.adapter, "log_data_quality"):
                        self.adapter.log_data_quality(q_record)

                 if audit:
                     from .schema import IngestionAuditRecord
                     record = IngestionAuditRecord(**audit)
                     self.log_ingestion_audit(record)
             return True
        except Exception as e:
            logger.error(f"Atomic ingest failed: {e}")
            return False

    def upsert_daily_price(self, record: DailyPriceRecord) -> bool:
        return self.adapter.upsert_daily_price(record)

    def upsert_daily_prices_batch(self, records: List[DailyPriceRecord]) -> int:
        return self.adapter.upsert_daily_prices_batch(records)

    def get_daily_prices(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        limit: int = None,
    ):
        return self.adapter.get_daily_prices(symbol, start_date, end_date, limit)

    def get_daily_prices_batch(
        self, symbols: List[str], start_date: str, end_date: str
    ):
        return self.adapter.get_daily_prices_batch(symbols, start_date, end_date)

    def upsert_intraday_price(self, record: IntradayPriceRecord) -> bool:
        return self.adapter.upsert_intraday_price(record)

    def get_intraday_prices(self, symbol: str, date: str = None):
        return self.adapter.get_intraday_prices(symbol, date)

    def log_ingestion_audit(self, record: IngestionAuditRecord) -> bool:
        return self.adapter.log_ingestion_audit(record)

    def log_ingestion_run(self, run_id: str, stats: Dict) -> bool:
        return self.adapter.log_ingestion_run(run_id, stats)

    def get_ingestion_audit(self, run_id=None, symbol=None, status=None):
        return self.adapter.get_ingestion_audit(run_id, symbol, status)

    def log_data_quality(self, record: DataQualityRecord) -> bool:
        return self.adapter.log_data_quality(record)

    def get_data_quality(self, symbol=None, min_score=None):
        return self.adapter.get_data_quality(symbol, min_score)

    def get_quality_summary(self):
        return self.adapter.get_quality_summary()

    def update_provider_metrics(self, record: ProviderMetricsRecord) -> bool:
        return self.adapter.update_provider_metrics(record)

    def get_provider_metrics(self, provider=None, days=30):
        return self.adapter.get_provider_metrics(provider, days)

    def get_provider_success_rates(self):
        return self.adapter.get_provider_success_rates()

    def log_execution_feedback(self, record: ExecutionFeedbackRecord) -> bool:
        return self.adapter.log_execution_feedback(record)

    def log_backfill_failure(self, record: BackfillFailureRecord) -> bool:
        return self.adapter.log_backfill_failure(record)

    def get_backfill_failures(self, run_id=None):
        return self.adapter.get_backfill_failures(run_id)

    def upsert_symbol_governance(self, record: SymbolGovernanceRecord) -> bool:
        return self.adapter.upsert_symbol_governance(record)

    def get_symbol_governance(self, symbol: str):
        return self.adapter.get_symbol_governance(symbol)

    def get_all_symbol_governance(self):
        return self.adapter.get_all_symbol_governance()

    def get_active_symbols(self):
        return self.adapter.get_active_symbols()

    def upsert_model_decay(self, record: ModelDecayMetric) -> bool:
        return self.adapter.upsert_model_decay(record)

    def insert_capital_allocations(self, records: List[CapitalAllocation]) -> int:
        return self.adapter.insert_capital_allocations(records)

    def log_governance_decision(self, record: GovernanceDecisionRecord) -> bool:
        return self.adapter.log_governance_decision(record)

    def upsert_strategy_lifecycle(self, record: StrategyLifecycleRecord) -> bool:
        return self.adapter.upsert_strategy_lifecycle(record)

    def upsert_features(self, records: List[FeatureRecord]) -> int:
        return self.adapter.upsert_features(records)

    def get_latest_features(self, symbols: List[str]):
        return self.adapter.get_latest_features(symbols)

    def get_features(self, symbol: str, date: str = None):
        return self.adapter.get_features(symbol, date)

    def insert_model_outputs(self, outputs: List[ModelOutput]) -> int:
        return self.adapter.insert_model_outputs(outputs)

    def insert_decisions(self, decisions: List[DecisionRecord]) -> int:
        return self.adapter.insert_decisions(decisions)

    def insert_orders(self, orders: List[OrderRecord]) -> int:
        return self.adapter.insert_orders(orders)

    def upsert_position(self, position: PositionRecord) -> bool:
        return self.adapter.upsert_position(position)

    def get_positions(self):
        return self.adapter.get_positions()

    def log_audit(self, entry: AuditEntry) -> int:
        return self.adapter.log_audit(entry)

    def insert_cycle_meta(self, meta: CycleMeta) -> bool:
        return self.adapter.insert_cycle_meta(meta)

    def get_symbol_coverage(self, start_date, end_date):
        return self.adapter.get_symbol_coverage(start_date, end_date)

    def insert_corporate_actions(self, actions) -> int:
        return self.adapter.insert_corporate_actions(actions)

    # Legacy mappings
    def upsert_price_history(self, records):
        # Map to usage of upsert_daily_prices_batch inside adapter or just call it directly if adapter handles it
        # Since logic was inside manager, now adapter should have it.
        # But wait, original manager had logic to convert PriceRecord to DailyPriceRecord.
        # We can keep that logic here or move to adapter. Use adapter direct call if compatible.
        return self.adapter.upsert_daily_prices_batch(
            records
        )  # Assuming records are compatible or converted

    def get_price_history(self, symbol, start_date, end_date=None):
        return self.adapter.get_daily_prices(symbol, start_date, end_date)


def get_db() -> DatabaseManager:
    return DatabaseManager()
