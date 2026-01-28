"""
data_intelligence/data_confidence.py

Data Confidence Memory Service (Ticket 2)

Maintains per-symbol, per-provider confidence scores using Bayesian updates.
Confidence affects capital allocation and provider selection.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger("DATA_CONFIDENCE")


@dataclass
class ConfidenceRecord:
    """Confidence record for a symbol-provider pair."""
    symbol: str
    provider: str
    confidence: float  # [0.0, 1.0]
    last_good_timestamp: str
    failure_rate_30d: float
    success_count_30d: int
    failure_count_30d: int
    avg_latency_ms: Optional[float]
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ConfidenceRecord":
        return cls(
            symbol=row['symbol'],
            provider=row['provider'],
            confidence=row.get('confidence', 0.5),
            last_good_timestamp=row.get('last_good_timestamp', ''),
            failure_rate_30d=row.get('failure_rate_30d', 0.0),
            success_count_30d=row.get('success_count_30d', 0),
            failure_count_30d=row.get('failure_count_30d', 0),
            avg_latency_ms=row.get('avg_latency_ms'),
            updated_at=row.get('updated_at', '')
        )


class DataConfidenceService:
    """
    Bayesian confidence tracking for data providers.

    Confidence is updated after each fetch:
    - Success: confidence increases (Bayesian update with success prior)
    - Failure: confidence decreases

    Confidence affects:
    - Provider selection (higher confidence = preferred)
    - Capital allocation (aggregate across providers)
    - Alert thresholds
    """

    # Bayesian update parameters
    PRIOR_ALPHA = 2  # Prior successes (optimistic start)
    PRIOR_BETA = 1   # Prior failures

    # Decay parameters
    DECAY_HALF_LIFE_DAYS = 14  # How fast old observations decay

    # Thresholds
    LOW_CONFIDENCE_THRESHOLD = 0.3
    HIGH_CONFIDENCE_THRESHOLD = 0.8

    def __init__(self, db_manager=None):
        """
        Initialize DataConfidenceService.

        Args:
            db_manager: DatabaseManager instance (lazy loaded if None)
        """
        self._db = db_manager
        self._cache: Dict[Tuple[str, str], ConfidenceRecord] = {}

    @property
    def db(self):
        if self._db is None:
            from database.manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    def get_confidence(self, symbol: str, provider: str) -> ConfidenceRecord:
        """
        Get confidence for a symbol-provider pair.

        Args:
            symbol: Stock symbol
            provider: Data provider name

        Returns:
            ConfidenceRecord
        """
        cache_key = (symbol, provider)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM data_confidence WHERE symbol = ? AND provider = ?",
                    (symbol, provider)
                )
                row = cursor.fetchone()
                if row:
                    record = ConfidenceRecord.from_row(dict(row))
                    self._cache[cache_key] = record
                    return record
        except Exception as e:
            logger.warning(f"Failed to get confidence for {symbol}/{provider}: {e}")

        # Return default
        return ConfidenceRecord(
            symbol=symbol,
            provider=provider,
            confidence=0.5,  # Neutral prior
            last_good_timestamp='',
            failure_rate_30d=0.0,
            success_count_30d=0,
            failure_count_30d=0,
            avg_latency_ms=None,
            updated_at=datetime.utcnow().isoformat() + 'Z'
        )

    def record_success(
        self,
        symbol: str,
        provider: str,
        latency_ms: Optional[float] = None
    ) -> ConfidenceRecord:
        """
        Record a successful data fetch.

        Args:
            symbol: Stock symbol
            provider: Data provider
            latency_ms: Fetch latency in milliseconds

        Returns:
            Updated ConfidenceRecord
        """
        current = self.get_confidence(symbol, provider)
        now = datetime.utcnow().isoformat() + 'Z'

        # Bayesian update: posterior = (α + successes) / (α + β + trials)
        new_success = current.success_count_30d + 1
        total = new_success + current.failure_count_30d
        new_confidence = (self.PRIOR_ALPHA + new_success) / (self.PRIOR_ALPHA + self.PRIOR_BETA + total)
        new_confidence = min(0.99, new_confidence)  # Cap at 0.99

        # Update latency (exponential moving average)
        if latency_ms is not None:
            if current.avg_latency_ms:
                new_latency = 0.7 * current.avg_latency_ms + 0.3 * latency_ms
            else:
                new_latency = latency_ms
        else:
            new_latency = current.avg_latency_ms

        # Update failure rate
        new_failure_rate = current.failure_count_30d / max(1, total)

        updated = ConfidenceRecord(
            symbol=symbol,
            provider=provider,
            confidence=round(new_confidence, 4),
            last_good_timestamp=now,
            failure_rate_30d=round(new_failure_rate, 4),
            success_count_30d=new_success,
            failure_count_30d=current.failure_count_30d,
            avg_latency_ms=round(new_latency, 2) if new_latency else None,
            updated_at=now
        )

        self._persist(updated)
        return updated

    def record_failure(
        self,
        symbol: str,
        provider: str,
        error_type: str = "UNKNOWN"
    ) -> ConfidenceRecord:
        """
        Record a failed data fetch.

        Args:
            symbol: Stock symbol
            provider: Data provider
            error_type: Type of error

        Returns:
            Updated ConfidenceRecord
        """
        current = self.get_confidence(symbol, provider)
        now = datetime.utcnow().isoformat() + 'Z'

        # Bayesian update with failure
        new_failure = current.failure_count_30d + 1
        total = current.success_count_30d + new_failure
        new_confidence = (self.PRIOR_ALPHA + current.success_count_30d) / (self.PRIOR_ALPHA + self.PRIOR_BETA + total)
        new_confidence = max(0.01, new_confidence)  # Floor at 0.01

        # Update failure rate
        new_failure_rate = new_failure / max(1, total)

        updated = ConfidenceRecord(
            symbol=symbol,
            provider=provider,
            confidence=round(new_confidence, 4),
            last_good_timestamp=current.last_good_timestamp,  # Keep last good
            failure_rate_30d=round(new_failure_rate, 4),
            success_count_30d=current.success_count_30d,
            failure_count_30d=new_failure,
            avg_latency_ms=current.avg_latency_ms,
            updated_at=now
        )

        self._persist(updated)

        # Log warning if confidence dropped below threshold
        if new_confidence < self.LOW_CONFIDENCE_THRESHOLD:
            logger.warning(json.dumps({
                "event": "LOW_CONFIDENCE_ALERT",
                "symbol": symbol,
                "provider": provider,
                "confidence": new_confidence,
                "failure_rate": new_failure_rate,
                "error_type": error_type
            }))

        return updated

    def _persist(self, record: ConfidenceRecord):
        """Persist confidence record to database."""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO data_confidence
                    (symbol, provider, confidence, last_good_timestamp, failure_rate_30d,
                     success_count_30d, failure_count_30d, avg_latency_ms, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.symbol,
                    record.provider,
                    record.confidence,
                    record.last_good_timestamp,
                    record.failure_rate_30d,
                    record.success_count_30d,
                    record.failure_count_30d,
                    record.avg_latency_ms,
                    record.updated_at
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist confidence for {record.symbol}/{record.provider}: {e}")

        # Update cache
        self._cache[(record.symbol, record.provider)] = record

    def get_best_provider(self, symbol: str, providers: List[str]) -> Optional[str]:
        """
        Get the best provider for a symbol based on confidence.

        Args:
            symbol: Stock symbol
            providers: List of available providers

        Returns:
            Provider name with highest confidence, or None
        """
        if not providers:
            return None

        best_provider = None
        best_confidence = -1.0

        for provider in providers:
            record = self.get_confidence(symbol, provider)
            if record.confidence > best_confidence:
                best_confidence = record.confidence
                best_provider = provider

        return best_provider

    def get_aggregate_confidence(self, symbol: str) -> float:
        """
        Get aggregate confidence across all providers for a symbol.

        Uses weighted average based on recency.
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM data_confidence WHERE symbol = ? ORDER BY updated_at DESC",
                    (symbol,)
                )
                rows = cursor.fetchall()

                if not rows:
                    return 0.5  # Neutral default

                # Weighted by confidence (higher confidence providers matter more)
                total_weight = 0.0
                weighted_sum = 0.0

                for row in rows:
                    confidence = row['confidence']
                    weight = confidence  # Self-weighted
                    weighted_sum += confidence * weight
                    total_weight += weight

                if total_weight > 0:
                    return round(weighted_sum / total_weight, 4)
                return 0.5

        except Exception as e:
            logger.error(f"Failed to get aggregate confidence for {symbol}: {e}")
            return 0.5

    def get_all_for_symbol(self, symbol: str) -> List[ConfidenceRecord]:
        """Get confidence records for all providers of a symbol."""
        records = []
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM data_confidence WHERE symbol = ?",
                    (symbol,)
                )
                for row in cursor.fetchall():
                    records.append(ConfidenceRecord.from_row(dict(row)))
        except Exception as e:
            logger.error(f"Failed to get confidence records for {symbol}: {e}")
        return records

    def get_provider_summary(self, provider: str) -> Dict[str, Any]:
        """Get summary statistics for a provider across all symbols."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as symbol_count,
                        AVG(confidence) as avg_confidence,
                        AVG(failure_rate_30d) as avg_failure_rate,
                        SUM(success_count_30d) as total_successes,
                        SUM(failure_count_30d) as total_failures,
                        AVG(avg_latency_ms) as avg_latency
                    FROM data_confidence
                    WHERE provider = ?
                """, (provider,))
                row = cursor.fetchone()

                if row:
                    return {
                        "provider": provider,
                        "symbol_count": row['symbol_count'] or 0,
                        "avg_confidence": round(row['avg_confidence'] or 0.5, 4),
                        "avg_failure_rate": round(row['avg_failure_rate'] or 0.0, 4),
                        "total_successes": row['total_successes'] or 0,
                        "total_failures": row['total_failures'] or 0,
                        "avg_latency_ms": round(row['avg_latency'] or 0, 2)
                    }
        except Exception as e:
            logger.error(f"Failed to get provider summary for {provider}: {e}")

        return {"provider": provider, "symbol_count": 0, "avg_confidence": 0.5}

    def decay_old_counts(self, days: int = 30):
        """
        Decay old success/failure counts.

        Should be run daily to ensure rolling 30-day window.
        """
        try:
            # Apply exponential decay to counts
            decay_factor = 0.5 ** (1 / self.DECAY_HALF_LIFE_DAYS)

            with self.db.get_connection() as conn:
                conn.execute("""
                    UPDATE data_confidence
                    SET success_count_30d = CAST(success_count_30d * ? AS INTEGER),
                        failure_count_30d = CAST(failure_count_30d * ? AS INTEGER),
                        updated_at = ?
                """, (decay_factor, decay_factor, datetime.utcnow().isoformat() + 'Z'))
                conn.commit()

            # Clear cache after decay
            self._cache.clear()

            logger.info(f"Applied decay factor {decay_factor:.4f} to all confidence counts")

        except Exception as e:
            logger.error(f"Failed to decay counts: {e}")


# Singleton instance
_instance: Optional[DataConfidenceService] = None


def get_data_confidence_service() -> DataConfidenceService:
    """Get singleton DataConfidenceService instance."""
    global _instance
    if _instance is None:
        _instance = DataConfidenceService()
    return _instance
