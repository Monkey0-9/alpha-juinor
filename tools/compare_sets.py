import json, sqlite3
from pathlib import Path

def load(fn):
    b = open(fn,'rb').read()
    for enc in ('utf-8','utf-16','utf-16-le','utf-16-be'):
        try:
            return json.loads(b.decode(enc))
        except Exception:
            pass
    raise RuntimeError('Failed to decode')

positions = load('data/alpaca_positions.json')
if isinstance(positions, dict):
    if 'positions' in positions:
        positions = positions['positions']
    else:
        for v in positions.values():
            if isinstance(v, list):
                positions = v
                break
alpaca_syms = {p['symbol'].upper() for p in positions}

conn = sqlite3.connect('data/nexus_audit.db')
cur = conn.cursor()
cur.execute("SELECT DISTINCT symbol FROM audit_log")
rows = cur.fetchall()
nexus_syms = {r[0].upper() for r in rows if r[0]}

print('Alpaca-only (<=20):')
for s in sorted(alpaca_syms - nexus_syms)[:20]:
    print(' -', s)
print('\nNexus-only (<=20):')
for s in sorted(nexus_syms - alpaca_syms)[:20]:
    print(' -', s)
print('\nCommon symbols:')
for s in sorted(alpaca_syms & nexus_syms):
    print(' -', s)
