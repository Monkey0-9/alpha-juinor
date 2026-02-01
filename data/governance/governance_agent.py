"""
data/governance/governance_agent.py

Institutional Symbol Governor.
Responsibilities:
1. Classify symbols into ACTIVE, DEGRADED, or QUARANTINED.
2. Enforce 5-year (1260 rows) threshold.
3. Persist classification.

OPTIMIZED: Batch queries instead of per-symbol queries.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List

from database.manager import DatabaseManager
from database.schema import SymbolGovernanceRecord

logger = logging.getLogger("SYMBOL_GOVERNOR")


class SymbolGovernor:
    """
    Manages symbol states based on absolute governance rules.
    """
    ACTIVE_THRESHOLD = 1260
    DEGRADED_THRESHOLD = 1000
    QUALITY_THRESHOLD = 0.6

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()

    def classify_all(self):
        """
        Perform a FAST governance sweep using batch queries.
        Optimized to avoid N+1 query problem.
        """
        with self.db.get_connection() as conn:
            # BATCH QUERY 1: Get row counts for all symbols
            cursor = conn.execute("""
                SELECT symbol, COUNT(*) as row_count
                FROM price_history
                GROUP BY symbol
            """)
            row_counts: Dict[str, int] = {row[0]: row[1] for row in cursor.fetchall()}

            # BATCH QUERY 2: Get latest quality scores
            cursor = conn.execute("""
                SELECT symbol, quality_score
                FROM data_quality
                WHERE rowid IN (
                    SELECT MAX(rowid) FROM data_quality GROUP BY symbol
                )
            """)
            quality_scores: Dict[str, float] = {}
            for row in cursor.fetchall():
                try:
                    quality_scores[row[0]] = float(row[1]) if row[1] else 0.0
                except (ValueError, TypeError):
                    quality_scores[row[0]] = 0.0

        symbols = list(row_counts.keys())
        logger.info(f"[GOVERNOR] Fast sweep of {len(symbols)} symbols...")

        # Track alerts (batch logging instead of per-symbol)
        low_quality_count = 0
        classified = {"ACTIVE": 0, "DEGRADED": 0, "QUARANTINED": 0}

        for symbol in symbols:
            row_count = row_counts.get(symbol, 0)
            quality_score = quality_scores.get(symbol, 0.0)

            # Check for low quality (only log first 50 to avoid spam)
            if quality_score < 0.75:
                low_quality_count += 1
                if low_quality_count <= 50:
                    print(
                        f"[MONITORING] ALERT: Low Data Quality for {symbol} "
                        f"({quality_score:.2f} < 0.75)"
                    )

            # Classify
            if row_count >= self.ACTIVE_THRESHOLD and quality_score >= self.QUALITY_THRESHOLD:
                state = "ACTIVE"
                reason = "Institutional Grade"
            elif row_count >= self.DEGRADED_THRESHOLD:
                state = "DEGRADED"
                reason = "Sub-optimal history or quality"
            else:
                state = "QUARANTINED"
                reason = f"Critical history failure: {row_count} rows"

            classified[state] += 1

            # Persist
            record = SymbolGovernanceRecord(
                symbol=symbol,
                history_rows=row_count,
                data_quality=quality_score,
                state=state,
                reason=reason,
                last_checked_ts=datetime.utcnow().isoformat(),
                metadata={},
            )
            try:
                self.db.upsert_symbol_governance(record)
            except AttributeError:
                pass  # Method missing, skip

        if low_quality_count > 50:
            print(f"[MONITORING] ... and {low_quality_count - 50} more low quality symbols")

        logger.info(
            f"[GOVERNOR] Completed: ACTIVE={classified['ACTIVE']}, "
            f"DEGRADED={classified['DEGRADED']}, QUARANTINED={classified['QUARANTINED']}"
        )

    def classify_symbol(self, symbol: str) -> str:
        """
        Classify a single symbol (for individual use).
        For bulk operations, use classify_all() instead.
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM price_history WHERE symbol = ?", (symbol,)
            )
            row = cursor.fetchone()
            row_count = row[0] if row else 0

            cursor = conn.execute(
                "SELECT quality_score FROM data_quality WHERE symbol = ? ORDER BY recorded_at DESC LIMIT 1",
                (symbol,),
            )
            q_row = cursor.fetchone()
            quality_score = float(q_row[0]) if q_row and q_row[0] else 0.0

        if row_count >= self.ACTIVE_THRESHOLD and quality_score >= self.QUALITY_THRESHOLD:
            state = "ACTIVE"
            reason = "Institutional Grade"
        elif row_count >= self.DEGRADED_THRESHOLD:
            state = "DEGRADED"
            reason = "Sub-optimal history or quality"
        else:
            state = "QUARANTINED"
            reason = f"Critical history failure: {row_count} rows"

        record = SymbolGovernanceRecord(
            symbol=symbol,
            history_rows=row_count,
            data_quality=quality_score,
            state=state,
            reason=reason,
            last_checked_ts=datetime.utcnow().isoformat(),
            metadata={},
        )
        try:
            self.db.upsert_symbol_governance(record)
        except AttributeError:
            pass

        return state
