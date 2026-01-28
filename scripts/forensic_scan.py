
import sqlite3
import json
import pandas as pd
from collections import Counter

DB_PATH = "runtime/audit.db"

def scan_latest_cycle():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("Scanning for latest cycle...")
    # 1. Get latest cycle ID (using ID for speed as it's auto-increment)
    cursor.execute("SELECT cycle_id FROM decisions ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    if not row:
        print("No cycles found.")
        conn.close()
        return

    cycle_id = row['cycle_id']
    print(f"Latest Cycle: {cycle_id}")

    # Get stats for this cycle
    cursor.execute("SELECT COUNT(*) as cnt, MIN(timestamp) as start_ts, MAX(timestamp) as end_ts FROM decisions WHERE cycle_id = ?", (cycle_id,))
    stats = cursor.fetchone()
    print(f"Timestamp: {stats['start_ts']} - {stats['end_ts']}")
    print(f"Total Decisions: {stats['cnt']}")

    # 2. Group by Decision
    print("\n--- Decision Breakdown ---")
    cursor.execute("SELECT final_decision, COUNT(*) as c FROM decisions WHERE cycle_id = ? GROUP BY final_decision", (cycle_id,))
    for r in cursor.fetchall():
        print(f"{r['final_decision']}: {r['c']}")

    # 3. Group by Reason
    print("\n--- Reasons Analysis ---")
    cursor.execute("SELECT final_decision, symbol, reason_codes, raw_traceback FROM decisions WHERE cycle_id = ?", (cycle_id,))
    reasons_counter = Counter()
    sample_symbols = {}

    rows = cursor.fetchall()
    print(f"Analyzing {len(rows)} records...")

    first_traceback_printed = False
    for r in rows:
        try:
            val = r['reason_codes']
            if not val:
                 reasons = []
            else:
                 reasons = json.loads(val)

            # Print traceback for first WORKER_CRASH
            if "WORKER_CRASH" in reasons and not first_traceback_printed:
                print("\n[TRACEBACK for WORKER_CRASH]")
                print(r['raw_traceback'])
                first_traceback_printed = True

            # Convert list to tuple for hashing
            # Prefix with decision
            reason_tuple = (r['final_decision'],) + tuple(reasons)
            reasons_counter[reason_tuple] += 1
            if reason_tuple not in sample_symbols:
                sample_symbols[reason_tuple] = r['symbol']
        except Exception as e:
            reasons_counter[f"PARSING_ERROR: {str(e)}"] += 1

    for reason_tuple, count in reasons_counter.most_common(20):
        decision = reason_tuple[0]
        reasons = reason_tuple[1:]
        print(f"{decision} | {count}x {reasons} (e.g. {sample_symbols.get(reason_tuple)})")

    conn.close()

if __name__ == "__main__":
    scan_latest_cycle()
