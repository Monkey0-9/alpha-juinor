import sqlite3
import json

conn = sqlite3.connect('runtime/institutional_trading.db')
with open('configs/universe.json') as f:
    universe = json.load(f)

db_symbols = set([r[0] for r in conn.execute('SELECT DISTINCT symbol FROM price_history').fetchall()])
universe_symbols = set(universe['active_tickers'])

missing = universe_symbols - db_symbols
print(f"Universe size: {len(universe_symbols)}")
print(f"DB symbols: {len(db_symbols)}")
print(f"Missing from DB: {len(missing)}")
if missing:
    print(f"Missing symbols: {sorted(missing)}")
