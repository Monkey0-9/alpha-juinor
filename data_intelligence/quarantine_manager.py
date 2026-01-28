#!/usr/bin/env python3
"""
Institutional Symbol Quarantine Manager

Responsibility:
Manage symbol states based on data quality, provider reliability, and manual overrides.
States: ACTIVE, DEGRADED, QUARANTINED, MANUAL_REVIEW.
Quarantined symbols = no capital, no trades.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='[QUARANTINE] %(message)s')
logger = logging.getLogger("QUARANTINE")

class QuarantineManager:
    """
    Automated symbol quarantine system.
    """
    def __init__(self):
        self.db = DatabaseManager()

    def assess_symbol(self, symbol: str) -> str:
        """Assess the state of a symbol based on recent data quality."""
        # Get latest quality score
        quality_records = self.db.get_data_quality(symbol=symbol)
        if not quality_records:
            return "MANUAL_REVIEW" # No data = manual check required

        latest = quality_records[0]
        score = latest.get('quality_score', 0.0)

        if score >= 0.9:
            return "ACTIVE"
        elif score >= 0.7:
            return "DEGRADED"
        elif score >= 0.6:
            return "MANUAL_REVIEW"
        else:
            return "QUARANTINED"

    def update_all_states(self):
        """Update states for the entire universe."""
        try:
            with open("configs/universe.json", "r") as f:
                universe = json.load(f)
            tickers = universe.get("active_tickers", [])
        except Exception as e:
            logger.error(f"Failed to load universe: {e}")
            return

        for tk in tickers:
            state = self.assess_symbol(tk)
            # Store in trading_eligibility table if it exists, or symbol_state
            # Using TradingEligibilityRecord from schema
            from database.schema import TradingEligibilityRecord

            # Get latest quality score for the record
            quality_records = self.db.get_data_quality(symbol=tk)
            score = quality_records[0].get('quality_score', 0.0) if quality_records else 0.0

            record = TradingEligibilityRecord(
                symbol=tk,
                tradable=(state in ["ACTIVE", "DEGRADED"]),
                trade_restrictions={"state": state},
                data_quality_score=score,
                provider_confidences={}, # Placeholder
                last_updated_ts=datetime.utcnow().isoformat()
            )
            self.db.upsert_trading_eligibility(record)

            if state in ["QUARANTINED", "MANUAL_REVIEW"]:
                logger.warning(f"Symbol {tk} marked as {state} | Capital Allocation: DISABLED")

    def is_tradable(self, symbol: str) -> bool:
        """Check if a symbol is currently tradable."""
        # Query trading_eligibility
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT tradable FROM trading_eligibility WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        if row:
            return bool(row['tradable'])
        return False

if __name__ == "__main__":
    qm = QuarantineManager()
    qm.update_all_states()
