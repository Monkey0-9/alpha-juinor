"""
data_intelligence/data_state_machine.py

Institutional Data State Machine (Ticket 1)
States: OK | DEGRADED_DATA | STALE_DATA | INVALID_DATA | FAILED_PROVIDER

This is the single source of truth for symbol data health.
All other components query this state before making decisions.
"""

import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger("DATA_STATE_MACHINE")


class DataState(str, Enum):
    """Canonical data states for institutional trading."""
    OK = "OK"
    DEGRADED_DATA = "DEGRADED_DATA"
    STALE_DATA = "STALE_DATA"
    INVALID_DATA = "INVALID_DATA"
    FAILED_PROVIDER = "FAILED_PROVIDER"
    UNKNOWN = "UNKNOWN"


@dataclass
class SymbolDataState:
    """Symbol data state record."""
    symbol: str
    state: DataState
    last_seen: str  # ISO timestamp of last data point
    last_good_ts: str  # ISO timestamp of last fully validated data
    failure_count_30d: int
    updated_at: str
    reason: Optional[str] = None
    provider: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['state'] = self.state.value if isinstance(self.state, DataState) else self.state
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymbolDataState":
        state_val = d.get('state', 'UNKNOWN')
        try:
            state = DataState(state_val)
        except ValueError:
            state = DataState.UNKNOWN
        return cls(
            symbol=d['symbol'],
            state=state,
            last_seen=d.get('last_seen', ''),
            last_good_ts=d.get('last_good_ts', ''),
            failure_count_30d=d.get('failure_count_30d', 0),
            updated_at=d.get('updated_at', ''),
            reason=d.get('reason'),
            provider=d.get('provider'),
            metadata=d.get('metadata')
        )


class DataStateMachine:
    """
    Institutional Data State Machine.

    Manages state transitions for symbol data health.
    All state changes are logged and auditable.

    Transition Rules:
    - OK: quality >= 0.9, fresh data, no provider issues
    - DEGRADED_DATA: 0.6 <= quality < 0.9
    - STALE_DATA: data older than staleness_threshold
    - INVALID_DATA: quality < 0.6 or validation failures
    - FAILED_PROVIDER: provider unreachable or N consecutive failures

    State affects:
    - OK: Full capital allocation allowed
    - DEGRADED_DATA: 50% capital cap
    - STALE_DATA: 25% capital cap, monitoring alert
    - INVALID_DATA: 0% capital, quarantine
    - FAILED_PROVIDER: 0% capital, failover to backup provider
    """

    # Thresholds
    QUALITY_OK_THRESHOLD = 0.9
    QUALITY_DEGRADED_THRESHOLD = 0.6
    STALENESS_HOURS = 24
    PROVIDER_FAILURE_THRESHOLD = 3  # Consecutive failures before FAILED_PROVIDER

    # Capital multipliers per state
    CAPITAL_MULTIPLIERS = {
        DataState.OK: 1.0,
        DataState.DEGRADED_DATA: 0.5,
        DataState.STALE_DATA: 0.25,
        DataState.INVALID_DATA: 0.0,
        DataState.FAILED_PROVIDER: 0.0,
        DataState.UNKNOWN: 0.0
    }

    def __init__(self, db_manager=None):
        """
        Initialize DataStateMachine.

        Args:
            db_manager: DatabaseManager instance (lazy loaded if None)
        """
        self._db = db_manager
        self._state_cache: Dict[str, SymbolDataState] = {}
        self._provider_failure_counts: Dict[str, int] = {}

    @property
    def db(self):
        if self._db is None:
            from database.manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    def get_state(self, symbol: str) -> SymbolDataState:
        """
        Get current state for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            SymbolDataState object
        """
        # Check cache first
        if symbol in self._state_cache:
            return self._state_cache[symbol]

        # Query database
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM symbol_data_state WHERE symbol = ?",
                    (symbol,)
                )
                row = cursor.fetchone()
                if row:
                    state = SymbolDataState(
                        symbol=row['symbol'],
                        state=DataState(row['state']),
                        last_seen=row['last_seen'] or '',
                        last_good_ts=row['last_good_ts'] or '',
                        failure_count_30d=row['failure_count_30d'] or 0,
                        updated_at=row['updated_at'] or '',
                        reason=row['reason'],
                        provider=row['provider'],
                        metadata=json.loads(row['metadata_json']) if row['metadata_json'] else None
                    )
                    self._state_cache[symbol] = state
                    return state
        except Exception as e:
            logger.warning(f"Failed to query state for {symbol}: {e}")

        # Default: UNKNOWN
        return SymbolDataState(
            symbol=symbol,
            state=DataState.UNKNOWN,
            last_seen='',
            last_good_ts='',
            failure_count_30d=0,
            updated_at=datetime.utcnow().isoformat() + 'Z',
            reason="No state record found"
        )

    def transition_state(
        self,
        symbol: str,
        new_state: DataState,
        reason: str,
        provider: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> SymbolDataState:
        """
        Transition symbol to a new state.

        All transitions are logged for audit.

        Args:
            symbol: Stock symbol
            new_state: Target state
            reason: Human-readable reason for transition
            provider: Data provider (if relevant)
            metadata: Additional context

        Returns:
            Updated SymbolDataState
        """
        current = self.get_state(symbol)
        old_state = current.state
        now = datetime.utcnow().isoformat() + 'Z'

        # Update state
        updated = SymbolDataState(
            symbol=symbol,
            state=new_state,
            last_seen=current.last_seen or now,
            last_good_ts=now if new_state == DataState.OK else current.last_good_ts,
            failure_count_30d=current.failure_count_30d + (1 if new_state in [DataState.INVALID_DATA, DataState.FAILED_PROVIDER] else 0),
            updated_at=now,
            reason=reason,
            provider=provider or current.provider,
            metadata=metadata
        )

        # Persist to database
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO symbol_data_state
                    (symbol, state, last_seen, last_good_ts, failure_count_30d, updated_at, reason, provider, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    updated.symbol,
                    updated.state.value,
                    updated.last_seen,
                    updated.last_good_ts,
                    updated.failure_count_30d,
                    updated.updated_at,
                    updated.reason,
                    updated.provider,
                    json.dumps(updated.metadata) if updated.metadata else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist state for {symbol}: {e}")

        # Update cache
        self._state_cache[symbol] = updated

        # Log transition
        if old_state != new_state:
            logger.info(json.dumps({
                "event": "STATE_TRANSITION",
                "symbol": symbol,
                "old_state": old_state.value if isinstance(old_state, DataState) else old_state,
                "new_state": new_state.value,
                "reason": reason,
                "provider": provider,
                "timestamp": now
            }))

        return updated

    def evaluate_and_transition(
        self,
        symbol: str,
        quality_score: float,
        data_timestamp: str,
        provider: str,
        validation_passed: bool = True
    ) -> SymbolDataState:
        """
        Evaluate data quality and automatically transition state.

        This is the main entry point for data ingestion pipelines.

        Args:
            symbol: Stock symbol
            quality_score: Data quality score [0, 1]
            data_timestamp: ISO timestamp of data
            provider: Data provider name
            validation_passed: Whether validation checks passed
        """
        now = datetime.utcnow()

        # Check staleness
        try:
            data_ts = datetime.fromisoformat(data_timestamp.replace('Z', '+00:00').replace('+00:00', ''))
            age_hours = (now - data_ts).total_seconds() / 3600
            is_stale = age_hours > self.STALENESS_HOURS
        except Exception:
            is_stale = True
            age_hours = -1

        # Determine target state
        if not validation_passed:
            new_state = DataState.INVALID_DATA
            reason = f"Validation failed (quality={quality_score:.2f})"
        elif is_stale:
            new_state = DataState.STALE_DATA
            reason = f"Data is {age_hours:.1f}h old (threshold={self.STALENESS_HOURS}h)"
        elif quality_score >= self.QUALITY_OK_THRESHOLD:
            new_state = DataState.OK
            reason = f"Quality={quality_score:.2f} >= {self.QUALITY_OK_THRESHOLD}"
        elif quality_score >= self.QUALITY_DEGRADED_THRESHOLD:
            new_state = DataState.DEGRADED_DATA
            reason = f"Quality={quality_score:.2f} in degraded range"
        else:
            new_state = DataState.INVALID_DATA
            reason = f"Quality={quality_score:.2f} < {self.QUALITY_DEGRADED_THRESHOLD}"

        return self.transition_state(
            symbol=symbol,
            new_state=new_state,
            reason=reason,
            provider=provider,
            metadata={
                "quality_score": quality_score,
                "data_timestamp": data_timestamp,
                "age_hours": age_hours,
                "validation_passed": validation_passed
            }
        )

    def record_provider_failure(self, symbol: str, provider: str, error: str) -> SymbolDataState:
        """
        Record a provider failure and potentially transition to FAILED_PROVIDER.

        Args:
            symbol: Stock symbol
            provider: Provider that failed
            error: Error message
        """
        key = f"{symbol}:{provider}"
        self._provider_failure_counts[key] = self._provider_failure_counts.get(key, 0) + 1

        if self._provider_failure_counts[key] >= self.PROVIDER_FAILURE_THRESHOLD:
            return self.transition_state(
                symbol=symbol,
                new_state=DataState.FAILED_PROVIDER,
                reason=f"Provider {provider} failed {self._provider_failure_counts[key]} times: {error}",
                provider=provider,
                metadata={"consecutive_failures": self._provider_failure_counts[key], "last_error": error}
            )
        else:
            # Just record the failure without state change
            current = self.get_state(symbol)
            logger.warning(f"Provider failure for {symbol} via {provider}: {error} (count={self._provider_failure_counts[key]})")
            return current

    def clear_provider_failures(self, symbol: str, provider: str):
        """Clear failure count on successful fetch."""
        key = f"{symbol}:{provider}"
        self._provider_failure_counts[key] = 0

    def get_capital_multiplier(self, symbol: str) -> float:
        """
        Get capital multiplier based on data state.

        This is used by PM Brain to scale position sizes.
        """
        state = self.get_state(symbol)
        return self.CAPITAL_MULTIPLIERS.get(state.state, 0.0)

    def get_all_states(self) -> List[SymbolDataState]:
        """Get states for all symbols."""
        states = []
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM symbol_data_state ORDER BY symbol")
                for row in cursor.fetchall():
                    states.append(SymbolDataState(
                        symbol=row['symbol'],
                        state=DataState(row['state']),
                        last_seen=row['last_seen'] or '',
                        last_good_ts=row['last_good_ts'] or '',
                        failure_count_30d=row['failure_count_30d'] or 0,
                        updated_at=row['updated_at'] or '',
                        reason=row['reason'],
                        provider=row['provider'],
                        metadata=json.loads(row['metadata_json']) if row['metadata_json'] else None
                    ))
        except Exception as e:
            logger.error(f"Failed to get all states: {e}")
        return states

    def get_symbols_by_state(self, state: DataState) -> List[str]:
        """Get all symbols in a specific state."""
        symbols = []
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT symbol FROM symbol_data_state WHERE state = ?",
                    (state.value,)
                )
                symbols = [row['symbol'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get symbols by state: {e}")
        return symbols

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get summary of data health across universe.

        Returns dict with counts per state and overall metrics.
        """
        summary = {
            "counts": {s.value: 0 for s in DataState},
            "total": 0,
            "ok_pct": 0.0,
            "degraded_pct": 0.0,
            "failed_pct": 0.0,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }

        states = self.get_all_states()
        summary["total"] = len(states)

        for s in states:
            summary["counts"][s.state.value] = summary["counts"].get(s.state.value, 0) + 1

        if summary["total"] > 0:
            summary["ok_pct"] = summary["counts"][DataState.OK.value] / summary["total"]
            summary["degraded_pct"] = summary["counts"][DataState.DEGRADED_DATA.value] / summary["total"]
            failed_count = (
                summary["counts"][DataState.INVALID_DATA.value] +
                summary["counts"][DataState.FAILED_PROVIDER.value]
            )
            summary["failed_pct"] = failed_count / summary["total"]

        return summary


# Singleton instance
_instance: Optional[DataStateMachine] = None


def get_data_state_machine() -> DataStateMachine:
    """Get singleton DataStateMachine instance."""
    global _instance
    if _instance is None:
        _instance = DataStateMachine()
    return _instance
