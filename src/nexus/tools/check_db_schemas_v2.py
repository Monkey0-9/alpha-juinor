import sqlite3
import os

ts = "20260121_091700"
db_paths = [
    r"C:\mini-quant-fund\runtime\institutional_trading.db",
    r"C:\mini-quant-fund\runtime\audit.db"
]
output_file = rf"C:\mini-quant-fund\runtime\agent_results\{ts}\db_schema.txt"

def check_schemas():
    with open(output_file, "w") as f:
        for db_path in db_paths:
            if not os.path.exists(db_path):
                f.write(f"Database not found: {db_path}\n")
                continue

            f.write(f"\n--- Schema for {os.path.basename(db_path)} ---\n")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            for table in tables:
                table_name = table[0]
                f.write(f"\nTable: {table_name}\n")
                columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
                for col in columns:
                    f.write(f"  {col[1]} ({col[2]})\n")

            conn.close()
    print(f"Schema written to {output_file}")

if __name__ == "__main__":
    check_schemas()
