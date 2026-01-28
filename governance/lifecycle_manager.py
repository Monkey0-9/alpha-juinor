
import logging
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import json
from typing import Optional, Dict, Any, List

from database.manager import DatabaseManager, SymbolGovernanceRecord
from data.collectors.data_router import DataRouter
from data.quality import compute_data_quality

logger = logging.getLogger(__name__)

class SymbolState(Enum):
    ACTIVE = "ACTIVE"
    QUARANTINED = "QUARANTINED"
    RETIRED = "RETIRED"
    DEGRADED = "DEGRADED"

class LifecycleManager:
    """
    Manages the lifecycle of trading symbols based on data quality and governance rules.

    States:
    - ACTIVE: Healthy, tradeable.
    - QUARANTINED: Data issues detected, trading halted.
    - RETIRED: Persistently bad -> removed from universe.
    """

    # Constants (Governance Policy)
    MIN_QUALITY_SCORE = 0.6
    QUARANTINE_TIMEOUT_DAYS = 30
    MIN_HISTORY_ROWS = 252 # 1 Year

    def __init__(self, db_manager: DatabaseManager, data_router: DataRouter):
        self.db = db_manager
        self.data_router = data_router

    def run_all(self, target_symbols: Optional[List[str]] = None):
        """Run lifecycle check for all known symbols or a targeted list."""
        if target_symbols:
            symbols = target_symbols
        else:
            # Get from governance table
            gov_records = self.db.get_all_symbol_governance()
            symbols = [r['symbol'] for r in gov_records]

        logger.info(f"[LIFECYCLE] Running periodic health checks for {len(symbols)} symbols...")
        for symbol in symbols:
            try:
                self.run_lifecycle_check(symbol)
            except Exception as e:
                logger.error(f"[LIFECYCLE] Failed check for {symbol}: {e}")

    def run_lifecycle_check(self, symbol: str) -> None:
        """
        Executes a single lifecycle check for a symbol.
        Transitions state if necessary.
        """
        # 1. Get Current State
        gov_record = self.db.get_symbol_governance(symbol)
        if not gov_record:
            # New symbol discovery (usually handled elsewhere, but safe default)
            current_state = SymbolState.QUARANTINED
            metadata = {}
        else:
            try:
                current_state = SymbolState(gov_record.get('state', 'QUARANTINED'))
            except ValueError:
                logger.warning(f"Invalid state for {symbol}: {gov_record.get('state')}. Resetting to QUARANTINED.")
                current_state = SymbolState.QUARANTINED

            metadata = gov_record.get('metadata', {})

        if current_state == SymbolState.RETIRED:
            return  # End of line

        # 2. Fetch Data & Check Quality
        # Mandatory: Load enough for history check (1260 days)
        df = self.data_router.get_daily_prices(symbol, limit=1300)

        # Institutional Fix: Ensure adjusted_close presence
        # If 'Adj Close' missing but 'Close' present, use 'Close' but mark as unadjusted
        if df is not None and not df.empty:
            if 'Adj Close' not in df.columns:
                 if 'Close' in df.columns:
                      df['Adj Close'] = df['Close']
                      metadata['adj_flag'] = 'using_raw_close'
                 elif 'close' in df.columns:
                      df['Adj Close'] = df['close']
                      metadata['adj_flag'] = 'using_raw_close'

        quality_score, reasons = compute_data_quality(df)
        history_len = len(df) if df is not None else 0

        # 3. Determine Health (Priority 0 requirement: 1260 rows)
        is_healthy = (
            quality_score >= self.MIN_QUALITY_SCORE and
            history_len >= 1260 # Institutional Mandate
        )

        new_state = current_state
        reason = gov_record.get('reason') if gov_record else "Initial Check"

        # 4. State Transitions
        if current_state == SymbolState.ACTIVE:
            if not is_healthy:
                new_state = SymbolState.QUARANTINED
                reason = f"Quality/History Drop: {quality_score:.2f} (Rows: {history_len}) (Reasons: {reasons})"
                metadata['quarantined_at'] = datetime.utcnow().isoformat()
                logger.warning(f"[LIFECYCLE] Quarantining {symbol}: {reason}")

        elif current_state == SymbolState.QUARANTINED:
            if is_healthy:
                new_state = SymbolState.ACTIVE
                reason = "Quality/History Restored"
                metadata.pop('quarantined_at', None) # Clear timestamp
                logger.info(f"[LIFECYCLE] Restoring {symbol}: {reason}")
            else:
                # Check Timeout
                q_at_str = metadata.get('quarantined_at')
                if q_at_str:
                    try:
                        q_at = datetime.fromisoformat(q_at_str)
                        days_in_jail = (datetime.utcnow() - q_at).days
                        if days_in_jail > self.QUARANTINE_TIMEOUT_DAYS:
                            new_state = SymbolState.RETIRED
                            reason = f"Timeout: Quarantined > {self.QUARANTINE_TIMEOUT_DAYS} days"
                            logger.error(f"[LIFECYCLE] RETIRING {symbol}: {reason}")
                    except:
                        metadata['quarantined_at'] = datetime.utcnow().isoformat()
                else:
                    # Mark start of quarantine if missing
                    if history_len > 0:
                        metadata['quarantined_at'] = datetime.utcnow().isoformat()

        # 5. Persist Changes
        record = SymbolGovernanceRecord(
            symbol=symbol,
            history_rows=history_len,
            data_quality=quality_score,
            state=new_state.value,
            reason=reason,
            last_checked_ts=datetime.utcnow().isoformat(),
            metadata=metadata
        )
        self.db.upsert_symbol_governance(record)
