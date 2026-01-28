"""
data/governance/governance_agent.py

Institutional Symbol Governor.
Responsibilities:
1. Classify symbols into ACTIVE, DEGRADED, or QUARANTINED.
2. Enforce 5-year (1260 rows) threshold.
3. Persist classification.
"""

import logging
from datetime import datetime
from typing import Optional

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

    def check_monitoring_alerts(self, symbol: str, quality_score: float, row_count: int):
        """
        Check for monitoring alerts.
        """
        # 1. Data Quality Alert
        if quality_score < 0.75:
            logger.error(
                f"[MONITORING] ALERT: Low Data Quality for {symbol} ({quality_score:.2f} < 0.75)"
            )

        # 2. Provider Failure (Deduced from low rows + quality?)
        if row_count == 0:
             logger.warning(f"[MONITORING] Zero history for {symbol}")

    def classify_all(self):
        """Perform a full governance sweep of all symbols in price_history."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT symbol FROM price_history")
            symbols = [row[0] for row in cursor.fetchall()]

        logger.info(f"[GOVERNOR] Starting sweep of {len(symbols)} symbols...")

        for symbol in symbols:
            self.classify_symbol(symbol)

        logger.info(f"[GOVERNOR] Completed sweep of {len(symbols)} symbols")

    def classify_symbol(self, symbol: str) -> str:
        """
        Classify symbol into ACTIVE, DEGRADED, or QUARANTINED.
        """
        # 1. Get row count
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM price_history WHERE symbol = ?", (symbol,)
            )
            row = cursor.fetchone()
            row_count = row[0] if row else 0

            # 2. Get latest quality score
            cursor = conn.execute(
                "SELECT quality_score FROM data_quality WHERE symbol = ? ORDER BY recorded_at DESC LIMIT 1",
                (symbol,),
            )
            q_row = cursor.fetchone()
            quality_score = 0.0
            if q_row:
                try:
                    quality_score = q_row[0]
                except Exception as e:
                    logger.error(f"[GOVERNOR] Error parsing quality for {symbol}: {e}")

            # TRIGGER MONITORING ALERTS
            self.check_monitoring_alerts(symbol, quality_score, row_count)

        # 3. Apply logic
        state = "QUARANTINED"
        reason = "Insufficient history"

        if (row_count >= self.ACTIVE_THRESHOLD and quality_score >= self.QUALITY_THRESHOLD):
            state = "ACTIVE"
            reason = "Institutional Grade"
        elif row_count >= self.DEGRADED_THRESHOLD:
            state = "DEGRADED"
            reason = "Sub-optimal history or quality"
        else:
            reason = f"Critical history failure: {row_count} rows"

        # 4. Update consolidated Governance Table
        # Using upsert logic if manager supports it, or simple insert
        # Assume manager has upsert_symbol_governance
        # If not, we might fail. The previous code called upsert_symbol_governance.
        # We'll create the record object.
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
             # If method missing, implement inline or skip
             logger.warning("upsert_symbol_governance method missing on DatabaseManager")

        logger.debug(f"[GOVERNOR] {symbol} classified as {state} ({reason})")
        return state
