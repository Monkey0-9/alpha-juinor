import json
import sqlite3
from pathlib import Path


def load_json(fn):
    b = open(fn, 'rb').read()
    for enc in ('utf-8', 'utf-16', 'utf-16-le', 'utf-16-be'):
        try:
            return json.loads(b.decode(enc))
        except Exception:
            pass
    raise RuntimeError('Failed to decode ' + fn)


def main():
    pos_file = Path('data/alpaca_positions.json')
    acct_file = Path('data/alpaca_account.json')
    db_file = Path('data/nexus_audit.db')

    if not pos_file.exists() or not acct_file.exists():
        print('Missing Alpaca JSON files in data/. Run the fetch step first.')
        return

    positions = load_json(str(pos_file))
    # positions may be a dict wrapper or a list — normalize to list of dicts
    if isinstance(positions, dict):
        if 'positions' in positions and isinstance(positions['positions'], list):
            positions = positions['positions']
        else:
            # Some API responses may return a single object or mapping; try to coerce
            try:
                positions = list(positions.values())
            except Exception:
                positions = [positions]
    acct = load_json(str(acct_file))

    print('Alpaca account status:', acct.get('status'))
    print('Cash:', acct.get('cash'), 'Equity:', acct.get('equity'), 'BuyingPower:', acct.get('buying_power'))
    print('Alpaca positions count:', len(positions))

    # load audit db
    if not db_file.exists():
        print('No Nexus audit DB at', db_file)
        return

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    # latest audits per symbol
    cur.execute("SELECT symbol, status, details, timestamp FROM audit_log ORDER BY timestamp DESC")
    audits = cur.fetchall()
    latest_audit = {}
    for sym, status, details, ts in audits:
        if sym not in latest_audit:
            latest_audit[sym] = {'status': status, 'details': details, 'timestamp': ts}

    # latest trades per symbol
    cur.execute("SELECT symbol, side, qty, price, status, timestamp FROM trade_history ORDER BY timestamp DESC")
    trades = cur.fetchall()
    latest_trade = {}
    for sym, side, qty, price, status, ts in trades:
        if sym not in latest_trade:
            latest_trade[sym] = {'side': side, 'qty': qty, 'price': price, 'status': status, 'timestamp': ts}

    # Build a set of symbols
    alpaca_syms = {p['symbol'].upper() for p in positions}
    nexus_syms = set(list(latest_audit.keys()) + list(latest_trade.keys()))

    print('\nSymbols in Alpaca positions but not in Nexus audit/trades:')
    for sym in sorted(alpaca_syms - nexus_syms):
        print('  -', sym)

    print('\nSymbols in Nexus audit/trades but not in Alpaca positions:')
    for sym in sorted(nexus_syms - alpaca_syms):
        print('  -', sym)

    print('\nSummary for matching symbols:')
    for sym in sorted(alpaca_syms & nexus_syms):
        pos = next((p for p in positions if p['symbol'].upper() == sym), None)
        audit = latest_audit.get(sym)
        trade = latest_trade.get(sym)
        qty = pos.get('qty') if pos else None
        mv = pos.get('market_value') if pos else None
        print(f'\nSymbol: {sym}')
        print('  Alpaca -> qty:', qty, 'market_value:', mv)
        if audit:
            print('  Nexus Audit ->', audit['status'], audit['details'])
        if trade:
            print('  Nexus Last Trade ->', trade['side'], trade['qty'], trade['price'], trade['status'])

    # overall observation
    print('\nObservations:')
    if float(acct.get('buying_power', 0) or 0) == 0:
        print(' - Alpaca account reports BuyingPower=0 (no available buying power)')
    if float(acct.get('cash', 0) or 0) < 0:
        print(' - Alpaca account cash negative:', acct.get('cash'))

    print('\nActions you can take:')
    print('  1) Update Nexus to accept Alpaca positions (import)')
    print('  2) Ask Nexus to liquidate positions to match Nexus internal state')
    print('  3) Adjust governance limits (e.g., increase NEXUS_MAX_POSITION_SIZE) and allow new trades')
    print('  4) Manual review and selective sync')

if __name__ == "__main__":
    main()
