#!/usr/bin/env python3
"""
HUGE FUNDS - LIVE TRADE MONITOR
=================================
Real-time terminal dashboard for Alpaca paper trading.
Shows account, positions, orders, and ALL trades in terminal.

Usage: python live_monitor.py
"""

import time
import json
import sys
import os
from datetime import datetime
from collections import deque

# Try to import requests, fallback to urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import urllib.error

BASE_URL = "http://localhost:8000"
REFRESH_SECONDS = 2  # Faster refresh for real-time trades
MAX_TRADE_HISTORY = 50

# Windows-safe colors (no unicode)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[90m'
    END = '\033[0m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def api_get(endpoint):
    """Make GET request to backend"""
    url = f"{BASE_URL}{endpoint}"
    try:
        if HAS_REQUESTS:
            resp = requests.get(url, timeout=10)
            return resp.json() if resp.status_code == 200 else None
        else:
            req = urllib.request.Request(url, method='GET')
            resp = urllib.request.urlopen(req, timeout=10)
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {"error": str(e)}

def format_money(val):
    """Format currency value"""
    if val is None:
        return "N/A"
    try:
        v = float(val)
        return f"${v:,.2f}"
    except:
        return str(val)

def format_change(val):
    """Format with color for gain/loss"""
    if val is None:
        return "N/A"
    try:
        v = float(val)
        color = Colors.GREEN if v >= 0 else Colors.RED
        sign = "+" if v >= 0 else ""
        return f"{color}{sign}{v:,.2f}{Colors.END}"
    except:
        return str(val)

def print_header():
    print(f"{Colors.BOLD}{Colors.WHITE}")
    print("=" * 70)
    print("  HUGE FUNDS - LIVE TRADE MONITOR")
    print("  Top 1% Worldwide Elite Collective - Alpaca Paper Trading")
    print("=" * 70)
    print(f"{Colors.END}")
    print(f"{Colors.DIM}  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Refresh: {REFRESH_SECONDS}s | Ctrl+C to exit{Colors.END}")
    print()

def print_account(data):
    print(f"{Colors.BOLD}{Colors.BLUE}  [ACCOUNT]{Colors.END}")
    if data and "error" not in data:
        status = data.get('status', 'UNKNOWN')
        status_color = Colors.GREEN if status == 'ACTIVE' else Colors.YELLOW
        print(f"    Status:      {status_color}{status}{Colors.END}")
        print(f"    Account:     {data.get('account_number', 'N/A')}")
        print(f"    Buying Pwr:  {Colors.BOLD}{format_money(data.get('buying_power'))}{Colors.END}")
        print(f"    Cash:        {format_money(data.get('cash'))}")
        print(f"    Portfolio:   {format_money(data.get('portfolio_value'))}")
        print(f"    Equity:      {format_money(data.get('equity'))}")
    else:
        print(f"    {Colors.RED}ERROR: {data.get('error', 'Cannot connect to backend')}{Colors.END}")
    print()

def print_positions(data):
    print(f"{Colors.BOLD}{Colors.BLUE}  [POSITIONS]{Colors.END}")
    if data and isinstance(data, list) and len(data) > 0:
        # Header
        print(f"    {'Symbol':<8} {'Qty':>8} {'Price':>12} {'Market Val':>14} {'Unreal P/L':>14}")
        print(f"    {'-'*60}")
        total_value = 0.0
        total_pl = 0.0
        for pos in data:
            sym = pos.get('symbol', 'N/A')
            qty = pos.get('qty', '0')
            price = pos.get('current_price', pos.get('lastday_price', '0'))
            mkt_val = pos.get('market_value', '0')
            unreal = pos.get('unrealized_pl', '0')
            try:
                total_value += float(mkt_val) if mkt_val else 0
                total_pl += float(unreal) if unreal else 0
            except:
                pass
            pl_str = format_change(unreal)
            print(f"    {sym:<8} {qty:>8} {format_money(price):>12} {format_money(mkt_val):>14} {pl_str:>20}")
        print(f"    {'-'*60}")
        print(f"    {'TOTAL':<8} {'':>8} {'':>12} {Colors.BOLD}{format_money(total_value):>14}{Colors.END} {format_change(total_pl):>20}")
    elif data and isinstance(data, list):
        print(f"    {Colors.YELLOW}No open positions{Colors.END}")
    else:
        print(f"    {Colors.RED}ERROR: Cannot fetch positions{Colors.END}")
    print()

def print_orders(data):
    print(f"{Colors.BOLD}{Colors.BLUE}  [RECENT ORDERS]{Colors.END}")
    if data and isinstance(data, list) and len(data) > 0:
        # Show last 10 orders
        orders = data[:10]
        print(f"    {'Symbol':<8} {'Side':<6} {'Qty':>8} {'Type':<10} {'Status':<12} {'Submitted':>20}")
        print(f"    {'-'*75}")
        for o in orders:
            sym = o.get('symbol', 'N/A')
            side = o.get('side', 'N/A').upper()
            qty = o.get('qty', '0')
            otype = o.get('type', 'market')
            status = o.get('status', 'unknown')
            submitted = o.get('submitted_at', 'N/A')
            if submitted and len(submitted) > 19:
                submitted = submitted[:19]
            side_color = Colors.GREEN if side == 'BUY' else Colors.RED
            print(f"    {sym:<8} {side_color}{side:<6}{Colors.END} {qty:>8} {otype:<10} {status:<12} {submitted:>20}")
    elif data and isinstance(data, list):
        print(f"    {Colors.YELLOW}No orders yet{Colors.END}")
    else:
        print(f"    {Colors.RED}ERROR: Cannot fetch orders{Colors.END}")
    print()

def print_quick_commands():
    print(f"{Colors.DIM}")
    print("  Quick Commands:")
    print("    Buy AAPL:  curl -X POST \"http://localhost:8000/api/alpaca/buy?symbol=AAPL&qty=10\"")
    print("    Sell AAPL: curl -X POST \"http://localhost:8000/api/alpaca/sell?symbol=AAPL&qty=10\"")
    print("    Positions: curl http://localhost:8000/api/alpaca/positions")
    print("    Orders:    curl http://localhost:8000/api/alpaca/orders")
    print("    Kill All:  curl -X DELETE http://localhost:8000/api/alpaca/positions")
    print(f"{Colors.END}")

def print_trade_history(trade_history):
    print(f"{Colors.BOLD}{Colors.BLUE}  [LIVE TRADE HISTORY]{Colors.END}")
    if len(trade_history) > 0:
        print(f"    {'Time':<8} {'Symbol':<8} {'Side':<6} {'Qty':>8} {'Price':>12} {'Status':<12}")
        print(f"    {'-'*65}")
        for trade in list(trade_history)[-15:]:  # Show last 15 trades
            time_str = trade.get('time', 'N/A')[:8] if trade.get('time') else 'N/A'
            sym = trade.get('symbol', 'N/A')
            side = trade.get('side', 'N/A').upper()
            qty = trade.get('qty', '0')
            price = trade.get('price', '0')
            status = trade.get('status', 'unknown')
            side_color = Colors.GREEN if side == 'BUY' else Colors.RED
            print(f"    {time_str:<8} {sym:<8} {side_color}{side:<6}{Colors.END} {qty:>8} {format_money(price):>12} {status:<12}")
    else:
        print(f"    {Colors.YELLOW}No trades yet{Colors.END}")
    print()

def detect_new_orders(old_orders, new_orders):
    """Detect new orders that appeared"""
    if not old_orders:
        return new_orders
    
    old_ids = {o.get('id', o.get('order_id', '')) for o in old_orders}
    new_orders_list = [o for o in new_orders if o.get('id', o.get('order_id', '')) not in old_ids]
    return new_orders_list

def main():
    print("HugeFunds Live Monitor starting...")
    print("Connecting to backend at", BASE_URL)
    time.sleep(1)

    trade_history = deque(maxlen=MAX_TRADE_HISTORY)
    last_orders = []

    try:
        while True:
            clear_screen()
            print_header()

            # Fetch all data
            account = api_get("/api/alpaca/account")
            positions = api_get("/api/alpaca/positions")
            orders = api_get("/api/alpaca/orders")

            # Detect new orders
            new_orders = detect_new_orders(last_orders, orders) if orders else []
            if new_orders:
                for order in new_orders:
                    trade = {
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'symbol': order.get('symbol', 'N/A'),
                        'side': order.get('side', 'N/A'),
                        'qty': order.get('qty', '0'),
                        'price': order.get('filled_avg_price', order.get('limit_price', '0')),
                        'status': order.get('status', 'unknown')
                    }
                    trade_history.append(trade)
                    # Print immediate trade alert
                    side_color = Colors.GREEN if trade['side'].upper() == 'BUY' else Colors.RED
                    print(f"{Colors.BOLD}{side_color}[NEW TRADE]{Colors.END} {trade['time']} {trade['symbol']} {trade['side']} {trade['qty']} @ {format_money(trade['price'])} - {trade['status']}")
                    time.sleep(0.5)  # Brief pause to see the alert

            last_orders = orders if orders else []

            # Display
            print_account(account)
            print_positions(positions)
            print_orders(orders)
            print_trade_history(trade_history)
            print_quick_commands()

            # Countdown
            for i in range(REFRESH_SECONDS, 0, -1):
                print(f"\r  {Colors.DIM}Refreshing in {i}s...{Colors.END}", end='', flush=True)
                time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n\n  {Colors.YELLOW}Monitor stopped.{Colors.END}")
        sys.exit(0)

if __name__ == "__main__":
    main()
