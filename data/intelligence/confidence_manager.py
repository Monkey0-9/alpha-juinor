"""
data/intelligence/confidence_manager.py

Manages data quality confidence scoring and state transitions.
Implements the Data State Machine for intelligent degradation.
"""

import sqlite3
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class DataState(Enum):
    """Data quality states for intelligent degradation"""
    OK = "OK"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    INVALID = "INVALID"

class ConfidenceManager:
    """
    Manages data quality confidence and provider reliability.

    State Machine:
    - OK (confidence >= 0.8): Full sizing (100%)
    - DEGRADED (0.6-0.8): Reduced sizing (50%)
    - STALE (0.4-0.6): Minimal sizing (25%)
    - INVALID (< 0.4): Reject symbol
    """

    def __init__(self, db_path: str = "runtime/audit.db"):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        """Initialize confidence tracking tables"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Load schema
            import os
            schema_path = os.path.join('database', 'schema', 'data_confidence.sql')
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    conn.executescript(f.read())
                logger.info("[CONFIDENCE] Schema loaded successfully")
            else:
                logger.warning(f"Schema file not found: {schema_path}")

            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize confidence tables: {e}")

    def update_confidence(self, symbol: str, provider: str, score: float,
                         last_good_date: Optional[str] = None):
        """
        Update confidence score for a symbol.

        Args:
            symbol: Symbol to update
            provider: Data provider
            score: Confidence score (0.0 to 1.0)
            last_good_date: Last date with good data
        """
        state = self._score_to_state(score)

        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO data_confidence
                (symbol, provider, confidence_score, last_good_date, state)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    provider = excluded.provider,
                    confidence_score = excluded.confidence_score,
                    last_good_date = excluded.last_good_date,
                    state = excluded.state
            """, (symbol, provider, score, last_good_date or datetime.utcnow().isoformat(), state.value))
            conn.commit()
            conn.close()

            logger.info(f"[CONFIDENCE] {symbol}: score={score:.2f}, state={state.value}")
        except Exception as e:
            logger.error(f"Failed to update confidence for {symbol}: {e}")

    def get_confidence(self, symbol: str) -> Optional[Dict]:
        """Retrieve confidence record for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM data_confidence WHERE symbol = ?", (symbol,)
            )
            row = cursor.fetchone()
            conn.close()

            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get confidence for {symbol}: {e}")
            return None

    def get_sizing_multiplier(self, symbol: str) -> float:
        """
        Get position sizing multiplier based on confidence state.

        Returns:
            1.0 (OK), 0.5 (DEGRADED), 0.25 (STALE), 0.0 (INVALID)
        """
        record = self.get_confidence(symbol)
        if not record:
            # No record = assume OK but log warning
            logger.warning(f"[CONFIDENCE] No record for {symbol}, assuming OK")
            return 1.0

        state = DataState(record['state'])
        multipliers = {
            DataState.OK: 1.0,
            DataState.DEGRADED: 0.5,
            DataState.STALE: 0.25,
            DataState.INVALID: 0.0
        }

        return multipliers[state]

    def log_provider_failure(self, provider: str, symbol: Optional[str],
                            error_code: int, error_message: str,
                            failure_type: str = "TIMEOUT"):
        """Log a provider failure for tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO provider_failures
                (provider, symbol, error_code, error_message, failure_type)
                VALUES (?, ?, ?, ?, ?)
            """, (provider, symbol, error_code, error_message, failure_type))
            conn.commit()
            conn.close()

            # Update failure rate
            self._update_failure_rate(provider, symbol)

            logger.warning(f"[PROVIDER_FAILURE] {provider}/{symbol}: {error_code} - {error_message}")
        except Exception as e:
            logger.error(f"Failed to log provider failure: {e}")

    def _update_failure_rate(self, provider: str, symbol: Optional[str]):
        """Calculate and update 30-day failure rate"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Count failures in last 30 days
            cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
            cursor = conn.execute("""
                SELECT COUNT(*) FROM provider_failures
                WHERE provider = ? AND symbol = ? AND timestamp > ?
            """, (provider, symbol, cutoff))

            failure_count = cursor.fetchone()[0]

            # Assume 30 attempts per day (rough estimate)
            total_attempts = 30 * 30
            failure_rate = failure_count / total_attempts

            # Update confidence record
            if symbol:
                conn.execute("""
                    UPDATE data_confidence
                    SET failure_rate_30d = ?,
                        consecutive_failures = consecutive_failures + 1
                    WHERE symbol = ?
                """, (failure_rate, symbol))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update failure rate: {e}")

    def _score_to_state(self, score: float) -> DataState:
        """Convert confidence score to state"""
        if score >= 0.8:
            return DataState.OK
        elif score >= 0.6:
            return DataState.DEGRADED
        elif score >= 0.4:
            return DataState.STALE
        else:
            return DataState.INVALID

    def get_provider_stats(self, provider: str, days: int = 30) -> Dict:
        """Get provider reliability statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_failures,
                    COUNT(DISTINCT symbol) as affected_symbols,
                    AVG(CASE WHEN error_code = 403 THEN 1 ELSE 0 END) as entitlement_fail_rate
                FROM provider_failures
                WHERE provider = ? AND timestamp > ?
            """, (provider, cutoff))

            row = cursor.fetchone()
            conn.close()

            return {
                "provider": provider,
                "total_failures": row[0] or 0,
                "affected_symbols": row[1] or 0,
                "entitlement_fail_rate": row[2] or 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get provider stats: {e}")
            return {"provider": provider, "total_failures": 0}
