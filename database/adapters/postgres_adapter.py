"""
database/adapters/postgres_adapter.py
P1-5: PostgreSQL adapter for production deployment
"""
import os
import logging
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import pandas as pd

try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    ThreadedConnectionPool = None
    RealDictCursor = None

from .base import DatabaseAdapter
from ..errors import DatabaseError

logger = logging.getLogger(__name__)


class PostgresAdapter(DatabaseAdapter):
    """
    PostgreSQL implementation of DatabaseAdapter.

    Uses connection pooling for production-grade performance.

    Usage:
        export DATABASE_URL="postgresql://user:pass@localhost:5432/funddb"
        db = PostgresAdapter()
    """

    def __init__(self, database_url: Optional[str] = None):
        if not POSTGRES_AVAILABLE:
            raise ImportError(
                "psycopg2 not installed. Install with: pip install psycopg2-binary"
            )

        self.database_url = database_url or os.getenv("DATABASE_URL")

        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")

        # Create connection pool (min 2, max 20 connections)
        try:
            self.pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                dsn=self.database_url
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            raise DatabaseError(f"Failed to create Postgres connection pool: {e}")

    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager."""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Postgres transaction error: {e}")
            raise DatabaseError(f"Transaction failed: {e}")
        finally:
            if conn:
                self.pool.putconn(conn)

    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        with self.get_connection() as conn:
            yield conn

    def close(self):
        """Close all connections in pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM price_history;")
                price_count = cursor.fetchone()[0]

                return {
                    "healthy": True,
                    "engine": "postgres",
                    "version": version,
                    "price_history_rows": price_count
                }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    # Implement abstract methods with Postgres-specific SQL

    def get_daily_prices(self, symbol: str, start_date: str = None,
                        end_date: str = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Get daily prices for a symbol."""
        query = "SELECT * FROM price_history WHERE symbol = %s"
        params = [symbol]

        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)

        query += " ORDER BY date DESC"

        if limit:
            query += " LIMIT %s"
            params.append(limit)

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

            if not df.empty:
                df = df.sort_values("date")
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df

    def get_active_symbols(self) -> List[str]:
        """Get list of active symbols."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT symbol FROM symbol_governance WHERE state='ACTIVE'"
            )
            return [row[0] for row in cursor.fetchall()]

    # Stub implementations for other methods
    # In production, implement full Postgres-specific logic

    def upsert_daily_price(self, record) -> bool:
        """Upsert single daily price record."""
        # TODO: Implement Postgres-specific UPSERT (ON CONFLICT)
        return True

    def upsert_daily_prices_batch(self, records: List) -> int:
        """Batch upsert daily prices."""
        # TODO: Use Postgres COPY or batch INSERT for performance
        return len(records)

    def get_daily_prices_batch(self, symbols: List[str], start_date: str,
                               end_date: str) -> Dict[str, pd.DataFrame]:
        """Batch get daily prices."""
        # TODO: Implement efficient batch query
        return {}

    # Placeholder implementations for remaining methods
    def upsert_intraday_price(self, record) -> bool: return True
    def get_intraday_prices(self, symbol: str, date: str = None) -> pd.DataFrame: return pd.DataFrame()
    def log_ingestion_audit(self, record) -> bool: return True
    def log_ingestion_run(self, run_id: str, stats: Dict) -> bool: return True
    def get_ingestion_audit(self, **kwargs) -> List[Dict]: return []
    def log_data_quality(self, record) -> bool: return True
    def get_data_quality(self, **kwargs) -> List[Dict]: return []
    def get_quality_summary(self) -> Dict[str, Any]: return {}
    def update_provider_metrics(self, record) -> bool: return True
    def get_provider_metrics(self, **kwargs) -> List[Dict]: return []
    def get_provider_success_rates(self) -> Dict[str, float]: return {}
    def log_execution_feedback(self, record) -> bool: return True
    def log_backfill_failure(self, record) -> bool: return True
    def get_backfill_failures(self, **kwargs) -> List[Dict]: return []
    def upsert_symbol_governance(self, record) -> bool: return True
    def get_symbol_governance(self, symbol: str) -> Optional[Dict]: return None
    def get_all_symbol_governance(self) -> List[Dict]: return []
    def upsert_model_decay(self, record) -> bool: return True
    def insert_capital_allocations(self, records) -> int: return 0
    def log_governance_decision(self, record) -> bool: return True
    def upsert_strategy_lifecycle(self, record) -> bool: return True
    def upsert_features(self, records) -> int: return 0
    def get_latest_features(self, symbols: List[str]) -> Dict: return {}
    def get_features(self, symbol: str, date: str = None) -> Dict: return {}
    def insert_model_outputs(self, outputs) -> int: return 0
    def insert_decisions(self, decisions) -> int: return 0
    def insert_orders(self, orders) -> int: return 0
    def upsert_position(self, position) -> bool: return True
    def get_positions(self) -> List[Dict]: return []
    def log_audit(self, entry) -> int: return 1
    def insert_cycle_meta(self, meta) -> bool: return True
    def get_symbol_coverage(self, start_date, end_date) -> Dict: return {}
    def insert_corporate_actions(self, actions) -> int: return 0
