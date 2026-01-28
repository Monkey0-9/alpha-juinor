import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "runtime/audit.db"

def inspect_audit_db():
    print(f"[{datetime.now()}] Inspecting Audit DB: {DB_PATH}")
    try:
        conn = sqlite3.connect(DB_PATH)

        # 1. Check tables
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        print("\nTables found:")
        print(tables)

        # 2. Check decisions table
        if 'decisions' in tables['name'].values:
            count = pd.read_sql("SELECT COUNT(*) as count FROM decisions", conn).iloc[0]['count']
            print(f"\nTotal rows in 'decisions': {count}")

            if count > 0:
                print("\nMost recent 5 decisions:")
                recent = pd.read_sql("SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 5", conn)
                print(recent)
            else:
                print("\nNo decisions recorded yet.")
        else:
            print("\n'decisions' table NOT found!")

        conn.close()
    except Exception as e:
        print(f"Error inspecting DB: {e}")

if __name__ == "__main__":
    inspect_audit_db()
