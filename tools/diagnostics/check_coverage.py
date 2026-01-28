from database.manager import DatabaseManager
import json

db = DatabaseManager()
conn = db.get_connection()

# Check symbols with insufficient data
cur = conn.execute('''
    SELECT symbol, COUNT(*) as cnt
    FROM price_history
    GROUP BY symbol
    HAVING cnt < 1260
    ORDER BY cnt DESC
''')

insufficient = list(cur.fetchall())
print(f"Symbols with < 1260 rows: {len(insufficient)}")
if insufficient:
    print("\nTop 20 symbols needing more data:")
    for symbol, cnt in insufficient[:20]:
        print(f"  {symbol}: {cnt}/1260 rows ({100*cnt/1260:.1f}%)")

# Check symbols with >= 1260
cur2 = conn.execute('''
    SELECT COUNT(*)
    FROM (
        SELECT symbol, COUNT(*) as cnt
        FROM price_history
        GROUP BY symbol
        HAVING cnt >= 1260
    )
''')
sufficient = cur2.fetchone()[0]
print(f"\n✅ Symbols with >= 1260 rows: {sufficient}")

# Check universe.json
try:
    with open('configs/universe.json') as f:
        universe = json.load(f)
        total_universe = len(universe.get('active_tickers', []))
        print(f"📋 Total symbols in universe.json: {total_universe}")
        print(f"🎯 Completion: {sufficient}/{total_universe} ({100*sufficient/total_universe:.1f}%)")
except Exception as e:
    print(f"Could not load universe: {e}")
