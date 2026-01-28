"""
Prune Universe Configuration.
Aligns universe.json with Governance Reality (DB ACTIVE state).
"""
import json
import sqlite3
from pathlib import Path
from data.universe_manager import UnifiedUniverseManager

def main():
    db_path = "runtime/institutional_trading.db"
    universe_path = "configs/universe.json"

    # 1. Get Universe Config
    u_mgr = UnifiedUniverseManager(universe_path)
    target_tickers = u_mgr.get_active_universe()
    print(f"Target Universe Size: {len(target_tickers)}")

    # 2. Get DB Active
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM symbol_governance WHERE state='ACTIVE'")
    active_tickers = {row[0] for row in cursor.fetchall()}
    print(f"DB ACTIVE Count: {len(active_tickers)}")

    # 3. Identify Failures
    failures = [t for t in target_tickers if t not in active_tickers]
    print(f"Failed Symbols: {len(failures)}")

    if not failures:
        print("Universe is fully compliant. No pruning needed.")
        return

    # 4. Diagnose
    print("\n--- Failure Diagnosis ---")
    for sym in failures:
        cursor.execute("SELECT status, reason_code, error_message FROM ingestion_audit WHERE symbol=? ORDER BY finished_at DESC LIMIT 1", (sym,))
        audit = cursor.fetchone()
        if audit:
            print(f"{sym}: {audit[0]} | {audit[1]} | {audit[2]}")
        else:
            print(f"{sym}: NO AUDIT RECORD (Possible skipped?)")

    # 5. Prune
    valid_universe = [t for t in target_tickers if t in active_tickers]
    print(f"\nPruning {len(failures)} symbols. New Size: {len(valid_universe)}")

    with open(universe_path, 'r') as f:
        data = json.load(f)

    data['active_tickers'] = valid_universe
    data['last_pruned'] = "2026-01-22"

    with open(universe_path, 'w') as f:
        json.dump(data, f, indent=4)

    print("universe.json updated.")

if __name__ == "__main__":
    main()
