
import sqlite3
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = Path("runtime/institutional_trading.db")

MIN_HISTORY_ROWS = 1260
MIN_DATA_QUALITY = 0.6

def audit_data():
    if not DB_PATH.exists():
        logger.error(f"Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    logger.info("PHASE 1: Scanning Database Row Counts...")

    # 1. Get Counts
    try:
        cursor.execute("SELECT symbol, COUNT(*) as count FROM price_history GROUP BY symbol")
        counts = {row['symbol']: row['count'] for row in cursor.fetchall()}
    except Exception as e:
        logger.error(f"Failed to query price_history: {e}")
        conn.close()
        return

    # 2. Get Eligibility
    eligibility = {}
    try:
        cursor.execute("SELECT symbol, status, data_quality_score FROM trading_eligibility")
        for row in cursor.fetchall():
            eligibility[row['symbol']] = {
                'status': row['status'],
                'quality': row['data_quality_score'] if row['data_quality_score'] is not None else 0.0
            }
    except Exception as e:
        logger.error(f"Failed to query trading_eligibility: {e}")

    # 3. Merge key sets
    all_symbols = set(counts.keys()) | set(eligibility.keys())

    active_candidates = []
    backfill_needed = []
    degraded = []
    quarantined = []

    updates = []

    print(f"\n{'SYMBOL':<15} | {'ROWS':<10} | {'STATUS':<12} | {'QUALITY':<8} | {'ACTION'}")
    print("-" * 75)

    for sym in sorted(all_symbols):
        count = counts.get(sym, 0)
        elig = eligibility.get(sym, {'status': 'UNKNOWN', 'quality': 0.0})
        status = elig['status']
        quality = float(elig['quality'])

        new_status = status
        action = "NONE"

        if count >= MIN_HISTORY_ROWS:
            if quality >= MIN_DATA_QUALITY:
                if status != 'ACTIVE':
                    new_status = 'ACTIVE'
                    action = "PROMOTE"
                active_candidates.append(sym)
            else:
                if status == 'ACTIVE':
                     new_status = 'DEGRADED'
                     action = "DEGRADE(Q)"
                degraded.append(sym)
        else:
            # Missing data
            if count == 0:
                if status != 'QUARANTINED':
                    new_status = 'QUARANTINED'
                    action = "QUARANTINE"
                quarantined.append(sym)
                backfill_needed.append(sym)
            else:
                if status == 'ACTIVE':
                    new_status = 'DEGRADED'
                    action = "DEGRADE(N)"
                backfill_needed.append(sym)
                degraded.append(sym)

        print(f"{sym:<15} | {count:<10} | {status:<12} | {quality:<8.2f} | {action}")

        if new_status != status:
            updates.append((new_status, sym))

    # Apply Updates
    if updates:
        logger.info(f"Updating status for {len(updates)} symbols...")
        cursor.executemany("UPDATE trading_eligibility SET status = ? WHERE symbol = ?", updates)
        conn.commit()

    conn.close()

    print("\nSUMMARY:")
    print(f"ACTIVE:      {len(active_candidates)}")
    print(f"BACKFILL:    {len(backfill_needed)}")
    print(f"DEGRADED:    {len(degraded)}")
    print(f"QUARANTINED: {len(quarantined)}")

    if backfill_needed:
        with open("required_backfill.txt", "w") as f:
            for sym in backfill_needed:
                f.write(f"{sym}\n")
        logger.info("Saved required_backfill.txt")

if __name__ == "__main__":
    audit_data()
