import json

with open('configs/universe.json', 'r') as f:
    universe = json.load(f)

# Remove RKA
original_count = len(universe['active_tickers'])
universe['active_tickers'] = [t for t in universe['active_tickers'] if t != 'RKA']
new_count = len(universe['active_tickers'])

with open('configs/universe.json', 'w') as f:
    json.dump(universe, f, indent=2)

print(f"Removed RKA from universe")
print(f"Original count: {original_count}")
print(f"New count: {new_count}")
