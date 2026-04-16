"""
tools/list_low_quality.py

Diagnostic tool to list symbols with low data quality scores.
"""
import sqlite3
import csv
import sys
import os

# Adjust path to DB
DB_PATH = "runtime/institutional_trading.db"
THRESH = 0.75

def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Connecting to {DB_PATH}...")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Check if table exists
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='symbol_governance'")
        if not cur.fetchone():
            print("Table 'symbol_governance' does not exist.")
            return

        # Query low quality symbols
        print(f"Querying symbols with data_quality < {THRESH}...")
        cur.execute(
            "SELECT symbol, data_quality, reason, state FROM symbol_governance WHERE data_quality < ?",
            (THRESH,)
        )
        rows = cur.fetchall()

        print(f"Found {len(rows)} low quality symbols.")

        # Display first 20
        print("\nFirst 20 Low Quality Symbols:")
        print(f"{'SYMBOL':<10} {'SCORE':<10} {'STATE':<15} {'REASON'}")
        print("-" * 60)
        for row in rows[:20]:
            print(f"{row[0]:<10} {row[1]:<10.2f} {row[3]:<15} {row[2]}")

        # Write to CSV
        csv_path = "low_quality.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "score", "reason", "state"])
            w.writerows(rows)

        print(f"\nFull list written to {csv_path}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    main()
