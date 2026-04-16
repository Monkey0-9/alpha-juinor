"""
PostgreSQL Database Adapter for Institutional Trading System.

Uses SQLAlchemy + psycopg2 for production PostgreSQL with TimescaleDB support.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal

import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

Base = declarative_base()


class PostgresManager:
    """
    PostgreSQL database manager for institutional trading.

    Features:
    - SQLAlchemy ORM for database operations
    - Connection pooling with QueuePool
    - TimescaleDB hypertable support for time-series data
    - Full implementation of all database operations
    """

    def __init__(self):
        """Initialize PostgreSQL connection."""
        self._engine = None
        self._session_factory = None
        self._initialized = False

        # Database configuration
        self._host = os.getenv("POSTGRES_HOST", "localhost")
        self._port = int(os.getenv("POSTGRES_PORT", "5432"))
        self._database = os.getenv("POSTGRES_DB", "quant_fund")
        self._user = os.getenv("POSTGRES_USER", "postgres")
        self._password = os.getenv("POSTGRES_PASSWORD", "")
        self._pool_size = int(os.getenv("POOL_SIZE", "5"))
        self._max_overflow = int(os.getenv("MAX_OVERFLOW", "10"))

    def _get_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return f"postgresql://{self._user}:{self._password}@{self._host}:{self._port}/{self._database}"

    def _init_engine(self):
        """Initialize SQLAlchemy engine."""
        if self._engine is None:
            conn_str = self._get_connection_string()
            self._engine = create_engine(
                conn_str,
                poolclass=QueuePool,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_pre_ping=True,
                echo=False
            )
            self._session_factory = sessionmaker(bind=self._engine)
            self._initialized = True
            logger.info(f"PostgreSQL engine initialized: {self._host}:{self._port}/{self._database}")

    @contextmanager
    def transaction(self) -> Session:
        """Context manager for database transactions."""
        self._init_engine()
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction error: {e}")
            raise
        finally:
            session.close()

    def health_check(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            self._init_engine()
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {
                "status": "healthy",
                "engine": "postgresql",
                "host": self._host,
                "port": self._port,
                "database": self._database
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "engine": "postgresql"
            }

    def close(self):
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._initialized = False

    # =========================================================================
    # PRICE HISTORY OPERATIONS
    # =========================================================================

    def upsert_daily_price(self, record: Dict) -> bool:
        """Insert or update daily price record."""
        self._init_engine()
        query = text("""
            INSERT INTO price_history (symbol, date, open, high, low, close, volume, adjusted_close, source, pulled_at)
            VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :adjusted_close, :source, :pulled_at)
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adjusted_close = EXCLUDED.adjusted_close,
                source = EXCLUDED.source,
                pulled_at = EXCLUDED.pulled_at
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    def upsert_daily_prices_batch(self, records: List[Dict]) -> int:
        """Batch upsert daily prices."""
        self._init_engine()
        query = text("""
            INSERT INTO price_history (symbol, date, open, high, low, close, volume, adjusted_close, source, pulled_at)
            VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :adjusted_close, :source, :pulled_at)
            ON CONFLICT (symbol, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adjusted_close = EXCLUDED.adjusted_close,
                source = EXCLUDED.source,
                pulled_at = EXCLUDED.pulled_at
        """)
        with self.transaction() as session:
            session.execute(query, records)
        return len(records)

    def get_daily_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get daily price history for a symbol."""
        self._init_engine()
        query = "SELECT * FROM price_history WHERE symbol = :symbol"
        params = {"symbol": symbol}

        if start_date:
            query += " AND date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY date ASC"
        if limit:
            query += f" LIMIT {limit}"

        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df

    def get_daily_prices_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Get daily prices for multiple symbols."""
        self._init_engine()
        placeholders = ','.join([f"'{s}'" for s in symbols])
        query = f"""
            SELECT * FROM price_history
            WHERE symbol IN ({placeholders})
            AND date BETWEEN :start_date AND :end_date
            ORDER BY symbol, date
        """
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"start_date": start_date, "end_date": end_date})
        return {"data": df, "symbols": symbols}

    def upsert_intraday_price(self, record: Dict) -> bool:
        """Insert or update intraday price record."""
        self._init_engine()
        query = text("""
            INSERT INTO price_history_intraday (symbol, timestamp, open, high, low, close, volume, timeframe, source)
            VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :timeframe, :source)
            ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    def get_intraday_prices(self, symbol: str, date: Optional[str] = None) -> pd.DataFrame:
        """Get intraday prices for a symbol."""
        self._init_engine()
        query = "SELECT * FROM price_history_intraday WHERE symbol = :symbol"
        params = {"symbol": symbol}

        if date:
            query += " AND timestamp::date = :date"
            params["date"] = date

        query += " ORDER BY timestamp ASC"

        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df

    # =========================================================================
    # INGESTION AUDIT OPERATIONS
    # =========================================================================

    def log_ingestion_audit(self, record: Dict) -> bool:
        """Log ingestion audit entry."""
        self._init_engine()
        query = text("""
            INSERT INTO ingestion_audit (symbol, date, provider, status, rows_fetched, duration_ms, error_message, raw_hash)
            VALUES (:symbol, :date, :provider, :status, :rows_fetched, :duration_ms, :error_message, :raw_hash)
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    def log_ingestion_run(self, run_id: str, stats: Dict) -> bool:
        """Log ingestion run statistics."""
        self._init_engine()
        query = text("""
            INSERT INTO ingestion_audit_runs (run_id, start_time, end_time, symbols_processed, rows_fetched, errors, status)
            VALUES (:run_id, :start_time, :end_time, :symbols_processed, :rows_fetched, :errors, :status)
        """)
        with self.transaction() as session:
            session.execute(query, {"run_id": run_id, **stats})
        return True

    def get_ingestion_audit(
        self,
        run_id: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None
    ) -> pd.DataFrame:
        """Get ingestion audit records."""
        self._init_engine()
        conditions = []
        params = {}

        if run_id:
            conditions.append("run_id = :run_id")
            params["run_id"] = run_id
        if symbol:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol
        if status:
            conditions.append("status = :status")
            params["status"] = status

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM ingestion_audit WHERE {where_clause} ORDER BY timestamp DESC LIMIT 1000"

        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df

    # =========================================================================
    # DATA QUALITY OPERATIONS
    # =========================================================================

    def log_data_quality(self, record: Dict) -> bool:
        """Log data quality assessment."""
        self._init_engine()
        query = text("""
            INSERT INTO data_quality_log (symbol, date, quality_score, completeness_score, accuracy_score, timeliness_score, issues_json)
            VALUES (:symbol, :date, :quality_score, :completeness_score, :accuracy_score, :timeliness_score, :issues_json)
            ON CONFLICT (symbol, date) DO UPDATE SET
                quality_score = EXCLUDED.quality_score,
                completeness_score = EXCLUDED.completeness_score,
                accuracy_score = EXCLUDED.accuracy_score,
                timeliness_score = EXCLUDED.timeliness_score,
                issues_json = EXCLUDED.issues_json
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    def get_data_quality(
        self,
        symbol: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> pd.DataFrame:
        """Get data quality records."""
        self._init_engine()
        conditions = []
        params = {}

        if symbol:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol
        if min_score:
            conditions.append("quality_score >= :min_score")
            params["min_score"] = min_score

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM data_quality_log WHERE {where_clause} ORDER BY date DESC"

        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df

    def get_quality_summary(self) -> Dict[str, Any]:
        """Get aggregate data quality summary."""
        self._init_engine()
        query = """
            SELECT
                COUNT(*) as total_records,
                AVG(quality_score) as avg_quality,
                MIN(quality_score) as min_quality,
                MAX(quality_score) as max_quality,
                COUNT(CASE WHEN quality_score < 0.6 THEN 1 END) as low_quality_count
            FROM data_quality_log
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        """
        with self._engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
        return {
            "total_records": result[0],
            "avg_quality": result[1],
            "min_quality": result[2],
            "max_quality": result[3],
            "low_quality_count": result[4]
        }

    # =========================================================================
    # PROVIDER METRICS OPERATIONS
    # =========================================================================

    def update_provider_metrics(self, record: Dict) -> bool:
        """Update provider metrics."""
        self._init_engine()
        query = text("""
            INSERT INTO provider_metrics (provider_name, date, pulls, successes, failures, avg_quality_score, avg_response_time_ms)
            VALUES (:provider_name, :date, :pulls, :successes, :failures, :avg_quality_score, :avg_response_time_ms)
            ON CONFLICT (provider_name, date) DO UPDATE SET
                pulls = EXCLUDED.pulls,
                successes = EXCLUDED.successes,
                failures = EXCLUDED.failures,
                avg_quality_score = EXCLUDED.avg_quality_score,
                avg_response_time_ms = EXCLUDED.avg_response_time_ms
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    def get_provider_metrics(
        self,
        provider: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """Get provider metrics."""
        self._init_engine()
        query = """
            SELECT * FROM provider_metrics
            WHERE date >= CURRENT_DATE - INTERVAL ':days days'
        """
        params = {"days": days}
        if provider:
            query += " AND provider_name = :provider"
            params["provider"] = provider
        query += " ORDER BY date DESC"

        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df

    def get_provider_success_rates(self) -> pd.DataFrame:
        """Get provider success rates summary."""
        self._init_engine()
        query = """
            SELECT
                provider_name,
                SUM(pulls) as total_pulls,
                SUM(successes) as total_successes,
                SUM(failures) as total_failures,
                CASE WHEN SUM(pulls) > 0 THEN SUM(successes)::float / SUM(pulls) ELSE 0 END as success_rate,
                AVG(avg_quality_score) as avg_quality,
                AVG(avg_response_time_ms) as avg_response_time
            FROM provider_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY provider_name
            ORDER BY success_rate DESC
        """
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df

    # =========================================================================
    # EXECUTION FEEDBACK OPERATIONS
    # =========================================================================

    def log_execution_feedback(self, record: Dict) -> bool:
        """Log execution feedback."""
        self._init_engine()
        query = text("""
            INSERT INTO execution_feedback (order_id, symbol, side, quantity, fill_price, execution_time_ms, slippage_bps, venue, status)
            VALUES (:order_id, :symbol, :side, :quantity, :fill_price, :execution_time_ms, :slippage_bps, :venue, :status)
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    # =========================================================================
    # BACKFILL FAILURE OPERATIONS
    # =========================================================================

    def log_backfill_failure(self, record: Dict) -> bool:
        """Log backfill failure."""
        self._init_engine()
        query = text("""
            INSERT INTO backfill_failures (symbol, date, provider, error_message, retry_count)
            VALUES (:symbol, :date, :provider, :error_message, :retry_count)
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    def get_backfill_failures(self, run_id: Optional[str] = None) -> pd.DataFrame:
        """Get backfill failure records."""
        self._init_engine()
        query = "SELECT * FROM backfill_failures ORDER BY timestamp DESC LIMIT 100"
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df

    # =========================================================================
    # SYMBOL GOVERNANCE OPERATIONS
    # =========================================================================

    def upsert_symbol_governance(self, record: Dict) -> bool:
        """Upsert symbol governance record."""
        self._init_engine()
        query = text("""
            INSERT INTO symbol_governance (symbol, state, classification, data_quality_score, history_complete, last_updated, reason_codes)
            VALUES (:symbol, :state, :classification, :data_quality_score, :history_complete, :last_updated, :reason_codes)
            ON CONFLICT (symbol) DO UPDATE SET
                state = EXCLUDED.state,
                classification = EXCLUDED.classification,
                data_quality_score = EXCLUDED.data_quality_score,
                history_complete = EXCLUDED.history_complete,
                last_updated = EXCLUDED.last_updated,
                reason_codes = EXCLUDED.reason_codes
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    def get_symbol_governance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get governance state for a specific symbol."""
        self._init_engine()
        query = "SELECT * FROM symbol_governance WHERE symbol = :symbol"
        with self._engine.connect() as conn:
            result = conn.execute(text(query), {"symbol": symbol}).fetchone()
        if result:
            return dict(result._mapping)
        return None

    def get_all_symbol_governance(self) -> pd.DataFrame:
        """Get all symbol governance records."""
        self._init_engine()
        query = "SELECT * FROM symbol_governance ORDER BY symbol"
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return df

    def get_active_symbols(self) -> List[str]:
        """Get all symbols with ACTIVE state."""
        self._init_engine()
        query = "SELECT symbol FROM symbol_governance WHERE state = 'ACTIVE'"
        with self._engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
        return [r[0] for r in result]

    # =========================================================================
    # MODEL DECAY OPERATIONS
    # =========================================================================

    def upsert_model_decay(self, record: Dict) -> bool:
        """Upsert model decay metrics."""
        self._init_engine()
        query = text("""
            INSERT INTO model_decay_metrics (model_name, symbol, date, drift_score, accuracy_decay, feature_drift, prediction_bias, recommendation)
            VALUES (:model_name, :symbol, :date, :drift_score, :accuracy_decay, :feature_drift, :prediction_bias, :recommendation)
            ON CONFLICT (model_name, symbol, date) DO UPDATE SET
                drift_score = EXCLUDED.drift_score,
                accuracy_decay = EXCLUDED.accuracy_decay,
                feature_drift = EXCLUDED.feature_drift,
                prediction_bias = EXCLUDED.prediction_bias,
                recommendation = EXCLUDED.recommendation
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    # =========================================================================
    # CAPITAL ALLOCATION OPERATIONS
    # =========================================================================

    def insert_capital_allocations(self, records: List[Dict]) -> int:
        """Insert capital allocation records."""
        self._init_engine()
        query = text("""
            INSERT INTO capital_allocations (symbol, date, target_weight, allocation_score, confidence, regime, reasoning)
            VALUES (:symbol, :date, :target_weight, :allocation_score, :confidence, :regime, :reasoning)
        """)
        with self.transaction() as session:
            session.execute(query, records)
        return len(records)

    # =========================================================================
    # GOVERNANCE DECISION OPERATIONS
    # =========================================================================

    def log_governance_decision(self, record: Dict) -> bool:
        """Log governance decision."""
        self._init_engine()
        query = text("""
            INSERT INTO governance_decisions (cycle_id, symbol, decision, reason_codes, risk_score, cvar_breach, quality_score, final_decision)
            VALUES (:cycle_id, :symbol, :decision, :reason_codes, :risk_score, :cvar_breach, :quality_score, :final_decision)
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    # =========================================================================
    # STRATEGY LIFECYCLE OPERATIONS
    # =========================================================================

    def upsert_strategy_lifecycle(self, record: Dict) -> bool:
        """Upsert strategy lifecycle record."""
        self._init_engine()
        query = text("""
            INSERT INTO strategy_lifecycle (strategy_name, date, state, capital_allocated, sharpe_ratio, max_drawdown, decay_score, recommendation)
            VALUES (:strategy_name, :date, :state, :capital_allocated, :sharpe_ratio, :max_drawdown, :decay_score, :recommendation)
            ON CONFLICT (strategy_name, date) DO UPDATE SET
                state = EXCLUDED.state,
                capital_allocated = EXCLUDED.capital_allocated,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                max_drawdown = EXCLUDED.max_drawdown,
                decay_score = EXCLUDED.decay_score,
                recommendation = EXCLUDED.recommendation
        """)
        with self.transaction() as session:
            session.execute(query, record)
        return True

    # =========================================================================
    # FEATURE OPERATIONS
    # =========================================================================

    def upsert_features(self, records: List[Dict]) -> int:
        """Insert or update feature records."""
        self._init_engine()
        query = text("""
            INSERT INTO features (symbol, date, version, feature_json, source)
            VALUES (:symbol, :date, :version, :feature_json, :source)
            ON CONFLICT (symbol, date) DO UPDATE SET
                version = EXCLUDED.version,
                feature_json = EXCLUDED.feature_json,
                source = EXCLUDED.source
        """)
        with self.transaction() as session:
            session.execute(query, records)
        return len(records)

    def get_latest_features(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get latest features for a list of symbols."""
        self._init_engine()
        placeholders = ','.join([f"'{s}'" for s in symbols])
        query = f"""
            SELECT f1.* FROM features f1
            INNER JOIN (
                SELECT symbol, MAX(date) as max_date
                FROM features
                WHERE symbol IN ({placeholders})
                GROUP BY symbol
            ) f2 ON f1.symbol = f2.symbol AND f1.date = f2.max_date
        """
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        result = {}
        for _, row in df.iterrows():
            result[row["symbol"]] = row.to_dict()
        return result

    def get_features(self, symbol: str, date: Optional[str] = None) -> pd.DataFrame:
        """Get features for a symbol."""
        self._init_engine()
        query = "SELECT * FROM features WHERE symbol = :symbol"
        params = {"symbol": symbol}

        if date:
            query += " AND date = :date"
            params["date"] = date

        query += " ORDER BY date DESC"

        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df

    # =========================================================================
    # MODEL OUTPUTS OPERATIONS
    # =========================================================================

    def insert_model_outputs(self, outputs: List[Dict]) -> int:
        """Insert model output records."""
        self._init_engine()
        query = text("""
            INSERT INTO model_outputs (model_name, symbol, date, signal, confidence, version, metadata_json)
            VALUES (:model_name, :symbol, :date, :signal, :confidence, :version, :metadata_json)
        """)
        with self.transaction() as session:
            session.execute(query, outputs)
        return len(outputs)

    # =========================================================================
    # DECISIONS OPERATIONS
    # =========================================================================

    def insert_decisions(self, decisions: List[Dict]) -> int:
        """Insert decision records."""
        self._init_engine()
        query = text("""
            INSERT INTO decisions (cycle_id, symbol, final_decision, reason_codes, confidence, risk_score, timestamp)
            VALUES (:cycle_id, :symbol, :final_decision, :reason_codes, :confidence, :risk_score, :timestamp)
        """)
        with self.transaction() as session:
            session.execute(query, decisions)
        return len(decisions)

    # =========================================================================
    # ORDER OPERATIONS
    # =========================================================================

    def insert_orders(self, orders: List[Dict]) -> int:
        """Insert order records."""
        self._init_engine()
        query = text("""
            INSERT INTO orders (order_id, symbol, side, quantity, order_type, limit_price, status, filled_qty, avg_price, timestamp, reason_codes, metadata_json)
            VALUES (:order_id, :symbol, :side, :quantity, :order_type, :limit_price, :status, :filled_qty, :avg_price, :timestamp, :reason_codes, :metadata_json)
        """)
        with self.transaction() as session:
            session.execute(query, orders)
        return len(orders)

    # =========================================================================
    # POSITION OPERATIONS
    # =========================================================================

    def upsert_position(self, position: Dict) -> bool:
        """Insert or update position."""
        self._init_engine()
        query = text("""
            INSERT INTO positions (symbol, quantity, avg_cost, market_value, unrealized_pnl, realized_pnl, side, timestamp)
            VALUES (:symbol, :quantity, :avg_cost, :market_value, :unrealized_pnl, :realized_pnl, :side, :timestamp)
            ON CONFLICT (symbol) DO UPDATE SET
                quantity = EXCLUDED.quantity,
                avg_cost = EXCLUDED.avg_cost,
                market_value = EXCLUDED.market_value,
                unrealized_pnl = EXCLUDED.unrealized_pnl,
                realized_pnl = EXCLUDED.realized_pnl,
                side = EXCLUDED.side,
                timestamp = EXCLUDED.timestamp
        """)
        with self.transaction() as session:
            session.execute(query, position)
        return True

    def get_positions(self) -> List[Dict]:
        """Get all current positions."""
        self._init_engine()
        query = "SELECT * FROM positions ORDER BY symbol"
        with self._engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
        return [dict(r._mapping) for r in result]

    # =========================================================================
    # AUDIT OPERATIONS
    # =========================================================================

    def log_audit(self, entry: Dict) -> int:
        """Log audit entry."""
        self._init_engine()
        query = text("""
            INSERT INTO audit_log (timestamp, stage, symbol, action, details_json, metadata_json)
            VALUES (:timestamp, :stage, :symbol, :action, :details_json, :metadata_json)
        """)
        with self.transaction() as session:
            session.execute(query, entry)
        return 1

    # =========================================================================
    # CYCLE META OPERATIONS
    # =========================================================================

    def insert_cycle_meta(self, meta: Dict) -> bool:
        """Insert cycle metadata."""
        self._init_engine()
        query = text("""
            INSERT INTO cycle_meta (cycle_id, start_time, end_time, symbols_processed, decisions_count, orders_count, status, error_message, duration_ms)
            VALUES (:cycle_id, :start_time, :end_time, :symbols_processed, :decisions_count, :orders_count, :status, :error_message, :duration_ms)
        """)
        with self.transaction() as session:
            session.execute(query, meta)
        return True

    # =========================================================================
    # COVERAGE OPERATIONS
    # =========================================================================

    def get_symbol_coverage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get data coverage summary."""
        self._init_engine()
        query = """
            SELECT
                COUNT(DISTINCT symbol) as total_symbols,
                COUNT(*) as total_rows,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM price_history
            WHERE date BETWEEN :start_date AND :end_date
        """
        with self._engine.connect() as conn:
            result = conn.execute(text(query), {"start_date": start_date, "end_date": end_date}).fetchone()
        return {
            "total_symbols": result[0],
            "total_rows": result[1],
            "min_date": str(result[2]),
            "max_date": str(result[3])
        }

    # =========================================================================
    # CORPORATE ACTIONS OPERATIONS
    # =========================================================================

    def insert_corporate_actions(self, actions: List[Dict]) -> int:
        """Insert corporate action records."""
        self._init_engine()
        query = text("""
            INSERT INTO corporate_actions (symbol, date, action_type, old_value, new_value, ratio, source)
            VALUES (:symbol, :date, :action_type, :old_value, :new_value, :ratio, :source)
        """)
        with self.transaction() as session:
            session.execute(query, actions)
        return len(actions)

