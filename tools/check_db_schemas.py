import sqlite3
import os

db_paths = [
    r"C:\mini-quant-fund\runtime\institutional_trading.db",
    r"C:\mini-quant-fund\runtime\audit.db"
]

def check_schemas():
    for db_path in db_paths:
        if not os.path.exists(db_path):
            print(f"Database not found: {db_path}")
            continue

        print(f"\n--- Schema for {os.path.basename(db_path)} ---")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
            for col in columns:
                print(f"  {col[1]} ({col[2]})")

        conn.close()

if __name__ == "__main__":
    check_schemas()
