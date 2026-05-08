import sqlite3
import os

db_path = "data/nexus_audit.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM audit_log LIMIT 10")
        rows = cursor.fetchall()
        print(f"Audit Logs (Last 10): {len(rows)}")
        for row in rows:
            print(row)
            
        cursor.execute("SELECT * FROM trade_history LIMIT 10")
        rows = cursor.fetchall()
        print(f"Trade History (Last 10): {len(rows)}")
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Error reading DB: {e}")
    finally:
        conn.close()
else:
    print("Database file not found.")
