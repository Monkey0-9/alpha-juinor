"""
tools/update_universe.py

Update symbol tiering and active status based on data quality/governance.
"""
import sys
import os
import json
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from database.manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UPDATE_UNIVERSE")

THRESH_TIER_1 = 0.75
THRESH_TIER_2 = 0.65

def main():
    logger.info("Connecting to database...")
    db = DatabaseManager()

    with db.get_connection() as conn:
        # Fetch all governance records
        cursor = conn.execute(
            "SELECT symbol, data_quality, history_rows FROM symbol_governance"
        )
        rows = cursor.fetchall()

    logger.info(f"Processing {len(rows)} symbols...")

    tier_counts = {"TIER_1": 0, "TIER_2": 0, "TIER_3": 0}
    active_symbols = []

    updates = []

    for row in rows:
        symbol, score, history = row
        score = float(score) if score is not None else 0.0
        history = int(history) if history is not None else 0

        # Determine Tier
        # TIER 1: High quality + Sufficient history (1260)
        if score >= THRESH_TIER_1 and history >= 1260:
            tier = "TIER_1"
            active = True
        # TIER 2: Medium quality OR slightly less history (1000)
        elif score >= THRESH_TIER_2 and history >= 1000:
            tier = "TIER_2"
            active = True # Consider active but maybe lower weight
        else:
            tier = "TIER_3"
            active = False

        tier_counts[tier] += 1
        if active:
            active_symbols.append(symbol)

        updates.append((tier, symbol))

    # Batch Update Tiers (Assuming symbol_governance has 'tier' column, or we use metadata)
    # Check if 'tier' column exists
    with db.get_connection() as conn:
        cursor = conn.execute("PRAGMA table_info(symbol_governance)")
        columns = [c[1] for c in cursor.fetchall()]
        has_tier_col = 'tier' in columns

    if has_tier_col:
        logger.info("Updating 'tier' column in symbol_governance...")
        with db.get_connection() as conn:
            conn.executemany(
                "UPDATE symbol_governance SET tier = ? WHERE symbol = ?",
                updates
            )
    else:
        logger.warning("'tier' column missing in symbol_governance. Skipping column update.")

    # Update Active Status based on Tier
    # Assuming 'state' column logic: ACTIVE vs QUARANTINED/DEGRADED
    # We update 'state' to ACTIVE for Tier 1/2, QUARANTINED for Tier 3
    logger.info("Updating 'state' based on tiers...")
    state_updates = []
    for tier, symbol in updates:
        new_state = "ACTIVE" if tier in ["TIER_1", "TIER_2"] else "QUARANTINED"
        state_updates.append((new_state, symbol))

    with db.get_connection() as conn:
        conn.executemany(
            "UPDATE symbol_governance SET state = ? WHERE symbol = ?",
            state_updates
        )

    # Output Summary
    logger.info("="*40)
    logger.info("UNIVERSE UPDATE COMPLETE")
    logger.info("="*40)
    logger.info(f"TIER 1 (Active, HQ): {tier_counts['TIER_1']}")
    logger.info(f"TIER 2 (Active, MQ): {tier_counts['TIER_2']}")
    logger.info(f"TIER 3 (Inactive):   {tier_counts['TIER_3']}")
    logger.info("-" * 40)
    logger.info(f"TOTAL ACTIVE SYMBOLS: {len(active_symbols)}")

    # Optional: Write active universe to JSON for easy loading
    try:
        with open("configs/universe.json", "w") as f:
            json.dump({
                "active_tickers": active_symbols,
                "generated_at": datetime.utcnow().isoformat(),
                "tier_counts": tier_counts
            }, f, indent=2)
        logger.info("Updated configs/universe.json")
    except Exception as e:
        logger.warning(f"Could not update configs/universe.json: {e}")

if __name__ == "__main__":
    from datetime import datetime
    main()
