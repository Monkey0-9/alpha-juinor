
"""
Execution Trace Tool.
Prints execution decisions for the last cycle logic.
"""

import argparse
import sys
import os
import sqlite3
import pandas as pd
import json

# Add project root
sys.path.append(os.getcwd())

from database.manager import get_db

def main():
    parser = argparse.ArgumentParser(description="Execution Trace Tool")
    parser.add_argument("--last-cycle", action="store_true", help="Trace last cycle decisions")
    args = parser.parse_args()

    if args.last_cycle:
        trace_last_cycle()

def trace_last_cycle():
    # Use audit DB
    db_path = "runtime/audit.db"
    if not os.path.exists(db_path):
        print(f"Audit DB not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Get latest cycle ID from decisions table itself
    # Since main.py doesn't write to cycle_meta table in audit.db (it might not exist there)
    try:
        cursor.execute("SELECT cycle_id, timestamp FROM decisions ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
    except Exception as e:
        print(f"Error querying audit DB: {e}")
        return

    if not row:
        print("No decision data found.")
        return

    cycle_id = row[0]
    print(f"Tracing Cycle: {cycle_id} ({row[1]})")
    print("-" * 100)
    print(f"{'SYMBOL':<10} | {'DECISION':<20} | {'REASONS':<30} | {'NOTIONAL':<10} | {'CONVICTION':<10}")
    print("-" * 100)

    # 2. Get decisions for this cycle
    cursor.execute("""
        SELECT symbol, final_decision, reason_codes, timestamp, conviction, order_data
        FROM decisions
        WHERE cycle_id = ?
    """, (cycle_id,))

    decisions = cursor.fetchall()

    count_executed = 0
    count_skipped = 0

    for d in decisions:
        symbol = d[0]
        decision = d[1]
        reasons_raw = d[2]
        timestamp = d[3]
        conviction = d[4]
        order_data = d[5]

        reasons = reasons_raw
        # Parse JSON if possible
        try:
             import json
             if reasons_raw and reasons_raw.startswith('['):
                 reasons = json.loads(reasons_raw)
        except:
             pass

        # Parse metadata/order data to find execution audit info if available
        notional = "N/A"

        # in main.py we sent execution details in the audit payload.
        # But write_audit stores arguments.
        # 'decisions' table schema in audit/decision_log.py has:
        # cycle_id, symbol, timestamp, ..., final_decision, reason_codes, order_data
        # We didn't store explicit 'notional' column in audit.db decisions table.
        # But maybe inside order_data if we hacked it?
        # Ideally we parse it.
        pass

        reasons_str = str(reasons)
        print(f"{symbol:<10} | {decision:<20} | {reasons_str[:30]:<30} | {notional:<10} | {conviction:<10.2f}")

        if decision == 'EXECUTE' or decision == 'BUY' or decision == 'SELL':
            count_executed += 1
        elif 'SKIP' in decision:
            count_skipped += 1

    print("-" * 100)
    print(f"Summary: Executed={count_executed}, Skipped={count_skipped}")

if __name__ == "__main__":
    main()
