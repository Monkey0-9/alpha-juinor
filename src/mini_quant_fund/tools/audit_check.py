
import sqlite3
import json

DB_PATH = 'runtime/audit.db'

def check_latest_decisions():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get count of MARKET_CLOSED
        cursor.execute("SELECT count(*) FROM decisions WHERE final_decision='HOLD' AND reason_codes LIKE '%MARKET_CLOSED%'")
        count = cursor.fetchone()[0]
        print(f"Total MARKET_CLOSED decisions found: {count}")

        # Get latest 10
        cursor.execute("""
            SELECT symbol, final_decision, reason_codes, timestamp
            FROM decisions
            ORDER BY id DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()
        print("\nLatest 10 Decisions:")
        for row in rows:
            print(f"[{row[3]}] {row[0]}: {row[1]} | Reason: {row[2]}")

        conn.close()
    except Exception as e:
        print(f"Error checking audit DB: {e}")

if __name__ == "__main__":
    check_latest_decisions()
