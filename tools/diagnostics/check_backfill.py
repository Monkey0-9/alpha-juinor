from database.manager import DatabaseManager

db = DatabaseManager()
conn = db.get_connection()
cur = conn.execute('''SELECT symbol, COUNT(*) as cnt
                      FROM price_history
                      WHERE symbol IN ("AAPL", "MSFT", "GOOG", "SPY", "TLT")
                      GROUP BY symbol''')

print("\nRecently Backfilled Symbols:")
for row in cur.fetchall():
    symbol, cnt = row
    print(f"  {symbol}: {cnt} rows")

# Also check total
cur2 = conn.execute('SELECT COUNT(DISTINCT symbol) as symbols, SUM(cnt) as total FROM (SELECT symbol, COUNT(*) as cnt FROM price_history GROUP BY symbol)')
row = cur2.fetchone()
print(f"\nOverall Database:")
print(f"  Total Symbols: {row[0]}")
print(f"  Total Rows: {row[1]}")
