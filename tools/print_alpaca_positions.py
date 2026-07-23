import json
from pathlib import Path

def load(fn):
    b = open(fn,'rb').read()
    for enc in ('utf-8','utf-16','utf-16-le','utf-16-be'):
        try:
            return json.loads(b.decode(enc))
        except Exception:
            pass
    raise RuntimeError('Failed to decode')

p = load('data/alpaca_positions.json')
# normalize
if isinstance(p, dict):
    if 'positions' in p:
        p = p['positions']
    else:
        # try to find list inside
        for v in p.values():
            if isinstance(v, list):
                p = v
                break

print('Total entries:', len(p))
for i, pos in enumerate(p[:50]):
    sym = pos.get('symbol') or pos.get('ticker') or pos.get('asset_id')
    qty = pos.get('qty')
    mv = pos.get('market_value')
    print(f"{i+1:02d}. {sym}  qty={qty}  mv={mv}")
