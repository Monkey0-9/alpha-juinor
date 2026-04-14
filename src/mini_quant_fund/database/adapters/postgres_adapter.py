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
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Use ON CONFLICT DO UPDATE for proper UPSERT
                upsert_query = """
                INSERT INTO price_history (
                    symbol, date, open, high, low, close, volume, adjusted_close,
                    created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                )
                ON CONFLICT (symbol, date)
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    adjusted_close = EXCLUDED.adjusted_close,
                    updated_at = NOW()
                """
                cursor.execute(upsert_query, (
                    record['symbol'],
                    record['date'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume'],
                    record.get('adjusted_close', record['close'])
                ))
                return True
        except Exception as e:
            logger.error(f"Failed to upsert daily price for {record.get('symbol')}: {e}")
            return False

    def upsert_daily_prices_batch(self, records: List) -> int:
        """Batch upsert daily prices using execute_values for high performance."""
        if not records:
            return 0

        from psycopg2.extras import execute_values
        
        upsert_query = """
        INSERT INTO price_history (
            symbol, date, open, high, low, close, volume, adjusted_close,
            created_at, updated_at
        ) VALUES %s
        ON CONFLICT (symbol, date)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            adjusted_close = EXCLUDED.adjusted_close,
            updated_at = NOW()
        """
        
        data = [
            (
                r['symbol'],
                r['date'],
                r['open'],
                r['high'],
                r['low'],
                r['close'],
                r['volume'],
                r.get('adjusted_close', r['close'])
            )
            for r in records
        ]

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                execute_values(cursor, upsert_query, data, template="(%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())")
                return len(records)
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            raise DatabaseError(f"Batch upsert failed: {e}")

    def get_daily_prices_batch(self, symbols: List[str], start_date: str,
                               end_date: str) -> Dict[str, pd.DataFrame]:
        """Batch get daily prices efficiently with single query."""
        if not symbols:
            return {}

        try:
            with self.get_connection() as conn:
                # Use ANY for efficient batch query
                query = """
                SELECT * FROM price_history
                WHERE symbol = ANY(%s)
                """
                params = [symbols]

                if start_date:
                    query += " AND date >= %s"
                    params.append(start_date)
                if end_date:
                    query += " AND date <= %s"
                    params.append(end_date)

                query += " ORDER BY symbol, date"

                df = pd.read_sql_query(query, conn, params=params)

                if df.empty:
                    return {}

                # Convert date column and sort
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(['symbol', 'date'])

                # Split by symbol
                result = {}
                for symbol in symbols:
                    symbol_df = df[df['symbol'] == symbol].copy()
                    if not symbol_df.empty:
                        symbol_df.set_index('date', inplace=True)
                        result[symbol] = symbol_df

                return result
        except Exception as e:
            logger.error(f"Failed to get batch daily prices: {e}")
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
    def upsert_features(self, records: List) -> int:
        """Batch upsert features."""
        if not records:
            return 0
        from psycopg2.extras import execute_values
        import json
        query = """
        INSERT INTO features (symbol, date, features_json, version, created_at)
        VALUES %s
        ON CONFLICT (symbol, date)
        DO UPDATE SET
            features_json = EXCLUDED.features_json,
            version = EXCLUDED.version,
            created_at = NOW()
        """
        data = [(r.symbol, r.date, json.dumps(r.features), r.version) for r in records]
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                execute_values(cursor, query, data, template="(%s, %s, %s, %s, NOW())")
                return len(records)
        except Exception as e:
            logger.error(f"Failed to upsert features: {e}")
            return 0

    def insert_model_outputs(self, outputs: List) -> int:
        """Batch insert model outputs."""
        if not outputs:
            return 0
        from psycopg2.extras import execute_values
        import json
        query = """
        INSERT INTO model_outputs (cycle_id, symbol, agent_name, mu, sigma, confidence, metadata_json)
        VALUES %s
        """
        data = [(o.cycle_id, o.symbol, o.agent_name, o.mu, o.sigma, o.confidence, json.dumps(o.metadata)) for o in outputs]
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                execute_values(cursor, query, data)
                return len(outputs)
        except Exception as e:
            logger.error(f"Failed to insert model outputs: {e}")
            return 0

    def insert_decisions(self, decisions: List) -> int:
        """Batch insert decisions."""
        if not decisions:
            return 0
        from psycopg2.extras import execute_values
        import json
        query = """
        INSERT INTO decisions (
            cycle_id, symbol, timestamp, final_decision, position_size,
            stop_loss, trailing_params_json, reason_codes_json, 
            data_quality_score, provider_confidence, mu_hat, sigma_hat, 
            conviction, metadata_json, created_at
        ) VALUES %s
        """
        data = [
            (
                d.cycle_id, d.symbol, datetime.utcnow().isoformat() + "Z",
                d.final_decision, d.position_size, d.stop_loss,
                json.dumps(d.trailing_params) if d.trailing_params else None,
                json.dumps(d.reason_codes), d.data_quality_score,
                d.provider_confidence, d.mu_hat, d.sigma_hat, d.conviction,
                json.dumps(d.metadata) if d.metadata else None
            )
            for d in decisions
        ]
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                execute_values(cursor, query, data, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())")
                return len(decisions)
        except Exception as e:
            logger.error(f"Failed to insert decisions: {e}")
            return 0

    def insert_orders(self, orders: List) -> int:
        """Batch insert orders."""
        if not orders:
            return 0
        from psycopg2.extras import execute_values
        import json
        query = """
        INSERT INTO orders (
            order_id, cycle_id, symbol, side, qty, price, order_type,
            time_in_force, status, commission, slippage, created_at
        ) VALUES %s
        """
        data = [
            (
                o.order_id, o.cycle_id, o.symbol, o.side, o.qty, o.price,
                o.order_type, o.time_in_force, o.status, o.commission, o.slippage,
                datetime.utcnow().isoformat() + "Z"
            )
            for o in orders
        ]
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                execute_values(cursor, query, data)
                return len(orders)
        except Exception as e:
            logger.error(f"Failed to insert orders: {e}")
            return 0

    def get_consecutive_skips(self) -> Dict[str, int]:
        """
        Calculate consecutive skips for all symbols using SQL window functions.
        Use ID DESC to ensure true chronological ordering.
        """
        query = """
        WITH recent_decisions AS (
            SELECT symbol, final_decision, 
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY id DESC) as rn
            FROM decisions
        ),
        skip_flags AS (
            SELECT symbol, rn,
                   CASE WHEN final_decision LIKE 'SKIP_%' OR final_decision = 'REJECT' THEN 1 ELSE 0 END as is_skip
            FROM recent_decisions
        ),
        first_non_skip AS (
            SELECT symbol, MIN(rn) as first_active_rn
            FROM skip_flags
            WHERE is_skip = 0
            GROUP BY symbol
        )
        SELECT s.symbol, 
               CASE 
                 WHEN f.first_active_rn IS NULL THEN COUNT(s.rn)
                 ELSE f.first_active_rn - 1
               END as skip_count
        FROM skip_flags s
        LEFT JOIN first_non_skip f ON s.symbol = f.symbol
        GROUP BY s.symbol, f.first_active_rn
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query)
                results = cursor.fetchall()
                return {row['symbol']: row['skip_count'] for row in results}
        except Exception as e:
            logger.error(f"Failed to get consecutive skips: {e}")
            return {}
    def upsert_position(self, position) -> bool: return True
    def get_positions(self) -> List[Dict]: return []
    def log_audit(self, entry) -> int: return 1
    def insert_cycle_meta(self, meta) -> bool: return True
    def get_symbol_coverage(self, start_date, end_date) -> Dict: return {}
    def insert_corporate_actions(self, actions) -> int: return 0
