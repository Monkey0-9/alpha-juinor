import sqlite3

DB = "runtime/institutional_trading.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]

print("Tables in database:")
for t in tables:
    print(f"  - {t}")

conn.close()

