#!/usr/bin/env python3
"""
Comprehensive Alpaca broker status check using existing infrastructure
"""
import os
import sys
from dotenv import load_dotenv
from brokers.alpaca_broker import AlpacaExecutionHandler

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")
base_url = os.getenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")

if not api_key or not secret_key:
    print("ERROR: Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")
    sys.exit(1)

print(f"\n{'='*70}")
print(f"ALPACA BROKER STATUS CHECK")
print(f"{'='*70}")
print(f"Endpoint: {base_url}")
print(f"Environment: {'PAPER' if 'paper' in base_url.lower() else 'LIVE'}")

try:
    handler = AlpacaExecutionHandler(api_key, secret_key, base_url)

    # Account status
    print(f"\n{'='*70}")
    print(f"ACCOUNT STATUS")
    print(f"{'='*70}")
    account = handler.get_account()
    print(f"Status: {account.get('status')}")
    print(f"Equity: ${float(account.get('equity', 0)):,.2f}")
    print(f"Cash: ${float(account.get('cash', 0)):,.2f}")
    print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
    print(f"Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")

    # Current positions
    print(f"\n{'='*70}")
    print(f"CURRENT POSITIONS")
    print(f"{'='*70}")
    positions = handler.get_positions()
    if not positions:
        print("No open positions")
    else:
        for symbol, qty in positions.items():
            print(f"  {symbol}: {qty:,.4f} shares")

    # Recent orders
    print(f"\n{'='*70}")
    print(f"RECENT ORDERS (Last 20)")
    print(f"{'='*70}")

    # Get all orders
    all_orders = handler.get_orders(status="all", limit=20)

    if not all_orders:
        print("No orders found")
    else:
        for i, order in enumerate(all_orders, 1):
            status = order.get('status', 'unknown')
            symbol = order.get('symbol', '?')
            side = order.get('side', '?')
            qty = order.get('qty', 0)
            filled_qty = order.get('filled_qty', 0)
            created = order.get('created_at', 'unknown')[:19]

            print(f"\n{i}. {symbol} - {side.upper()}")
            print(f"   Qty: {qty} | Filled: {filled_qty} | Status: {status}")
            print(f"   Created: {created}")

            if order.get('filled_avg_price'):
                print(f"   Avg Fill Price: ${float(order.get('filled_avg_price')):,.2f}")

    # Recent fills
    print(f"\n{'='*70}")
    print(f"RECENT FILLS (Last 10)")
    print(f"{'='*70}")

    try:
        fills = handler.get_activities(type="FILL")[:10]
        if not fills:
            print("No recent fills")
        else:
            for i, fill in enumerate(fills, 1):
                print(f"\n{i}. {fill.get('symbol')} - {fill.get('side').upper()}")
                print(f"   Qty: {fill.get('qty')} @ ${float(fill.get('price')):,.2f}")
                print(f"   Time: {fill.get('transaction_time', '')[:19]}")
    except:
        print("Could not fetch fill activities")

    print(f"\n{'='*70}\n")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
