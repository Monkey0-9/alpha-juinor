import json

def load(fn):
    b = open(fn,'rb').read()
    for enc in ('utf-8','utf-16','utf-16-le','utf-16-be'):
        try:
            return json.loads(b.decode(enc))
        except Exception:
            pass
    raise RuntimeError('Failed to decode')

p = load('data/alpaca_positions.json')
print('type', type(p), 'len', len(p))
for pos in p:
    print('---')
    for k,v in pos.items():
        print(k, ':', v)

print('\n--- Account ---')
a = load('data/alpaca_account.json')
for k,v in a.items():
    print(k, ':', v)
