
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from contextlib import contextmanager

from ..schema import (
    DailyPriceRecord, IntradayPriceRecord, IngestionAuditRecord,
    DataQualityRecord, ProviderMetricsRecord, ExecutionFeedbackRecord,
    BackfillFailureRecord, TradingEligibilityRecord, SymbolGovernanceRecord,
    ModelDecayMetric, CapitalAllocation, GovernanceDecisionRecord,
    StrategyLifecycleRecord, PriceRecord, CorporateAction, FeatureRecord,
    ModelOutput, DecisionRecord, OrderRecord, PositionRecord, AuditEntry,
    CycleMeta
)
from ..errors import DatabaseError

class DatabaseAdapter(ABC):
    """
    Abstract interface for database operations.
    Implementations: SQLiteAdapter, PostgresAdapter.
    """

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return database health status."""
        pass

    @abstractmethod
    @contextmanager
    def transaction(self):
        """Yields a transactional context."""
        pass

    @abstractmethod
    def close(self):
        """Close connection."""
        pass

    # --- Daily Price History ---
    @abstractmethod
    def upsert_daily_price(self, record: DailyPriceRecord) -> bool:
        pass

    @abstractmethod
    def upsert_daily_prices_batch(self, records: List[DailyPriceRecord]) -> int:
        pass

    @abstractmethod
    def get_daily_prices(self, symbol: str, start_date: str = None,
                         end_date: str = None, limit: Optional[int] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_daily_prices_batch(self, symbols: List[str], start_date: str,
                               end_date: str) -> Dict[str, pd.DataFrame]:
        pass

    # --- Intraday ---
    @abstractmethod
    def upsert_intraday_price(self, record: IntradayPriceRecord) -> bool:
        pass

    @abstractmethod
    def get_intraday_prices(self, symbol: str, date: str = None) -> pd.DataFrame:
        pass

    # --- Ingestion Audit ---
    @abstractmethod
    def log_ingestion_audit(self, record: IngestionAuditRecord) -> bool:
        pass

    @abstractmethod
    def log_ingestion_run(self, run_id: str, stats: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_ingestion_audit(self, run_id: str = None, symbol: str = None, status: str = None) -> List[Dict]:
        pass

    # --- Data Quality ---
    @abstractmethod
    def log_data_quality(self, record: DataQualityRecord) -> bool:
        pass

    @abstractmethod
    def get_data_quality(self, symbol: str = None, min_score: float = None) -> List[Dict]:
        pass

    @abstractmethod
    def get_quality_summary(self) -> Dict[str, Any]:
        pass

    # --- Provider Metrics ---
    @abstractmethod
    def update_provider_metrics(self, record: ProviderMetricsRecord) -> bool:
        pass

    @abstractmethod
    def get_provider_metrics(self, provider: str = None, days: int = 30) -> List[Dict]:
        pass

    @abstractmethod
    def get_provider_success_rates(self) -> Dict[str, float]:
        pass

    # --- Execution Feedback ---
    @abstractmethod
    def log_execution_feedback(self, record: ExecutionFeedbackRecord) -> bool:
        pass

    # --- Backfill Failures ---
    @abstractmethod
    def log_backfill_failure(self, record: BackfillFailureRecord) -> bool:
        pass

    @abstractmethod
    def get_backfill_failures(self, run_id: str = None) -> List[Dict]:
        pass

    # --- Governance ---
    @abstractmethod
    def upsert_symbol_governance(self, record: SymbolGovernanceRecord) -> bool:
        pass

    @abstractmethod
    def get_symbol_governance(self, symbol: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_all_symbol_governance(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_active_symbols(self) -> List[str]:
        pass

    @abstractmethod
    def upsert_model_decay(self, record: ModelDecayMetric) -> bool:
        pass

    @abstractmethod
    def insert_capital_allocations(self, records: List[CapitalAllocation]) -> int:
        pass

    @abstractmethod
    def log_governance_decision(self, record: GovernanceDecisionRecord) -> bool:
        pass

    @abstractmethod
    def upsert_strategy_lifecycle(self, record: StrategyLifecycleRecord) -> bool:
        pass

    # --- Features ---
    @abstractmethod
    def upsert_features(self, records: List[FeatureRecord]) -> int:
        pass

    @abstractmethod
    def get_latest_features(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def get_features(self, symbol: str, date: str = None) -> Dict[str, Any]:
        pass

    # --- Model Outputs ---
    @abstractmethod
    def insert_model_outputs(self, outputs: List[ModelOutput]) -> int:
        pass

    # --- Trading ---
    @abstractmethod
    def insert_decisions(self, decisions: List[DecisionRecord]) -> int:
        pass

    @abstractmethod
    def insert_orders(self, orders: List[OrderRecord]) -> int:
        pass

    @abstractmethod
    def upsert_position(self, position: PositionRecord) -> bool:
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        pass

    @abstractmethod
    def log_audit(self, entry: AuditEntry) -> int:
        pass

    @abstractmethod
    def insert_cycle_meta(self, meta: CycleMeta) -> bool:
        pass

    @abstractmethod
    def get_symbol_coverage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        pass

    # --- Corporate Actions ---
    @abstractmethod
    def insert_corporate_actions(self, actions: List[CorporateAction]) -> int:
        pass
