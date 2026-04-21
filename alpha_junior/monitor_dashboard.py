#!/usr/bin/env python3
"""
Alpha Junior - Real-Time Trading Dashboard
Shows live trades, P&L, and system status in terminal
"""

import requests
import time
import os
import sys
from datetime import datetime
import json

# Colors for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_account_info():
    """Get account information"""
    try:
        response = requests.get('http://localhost:5000/api/trading/account', timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_positions():
    """Get current positions"""
    try:
        response = requests.get('http://localhost:5000/api/trading/positions', timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_orders():
    """Get recent orders"""
    try:
        response = requests.get('http://localhost:5000/api/trading/orders?status=all', timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def format_money(value):
    """Format money with color"""
    if value > 0:
        return f"{Colors.GREEN}+${value:,.2f}{Colors.END}"
    elif value < 0:
        return f"{Colors.RED}-${abs(value):,.2f}{Colors.END}"
    else:
        return f"${value:,.2f}"

def draw_box(title, content, width=70):
    """Draw a box around content"""
    print(f"┌{'─' * width}┐")
    print(f"│{Colors.BOLD}{title:^{width}}{Colors.END}│")
    print(f"├{'─' * width}┤")
    for line in content.split('\n'):
        print(f"│ {line:<{width-2}} │")
    print(f"└{'─' * width}┘")

def display_dashboard():
    """Display the full dashboard"""
    clear_screen()
    
    # Header
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║            🤖 ALPHA JUNIOR - LIVE TRADING DASHBOARD               ║")
    print("║                  Automated Trading System v2.0                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  📅 {Colors.CYAN}{now}{Colors.END}\n")
    
    # Get data
    account = get_account_info()
    positions = get_positions()
    orders = get_orders()
    
    # Check if server is running
    if not account:
        print(f"  {Colors.RED}❌ Server not running!{Colors.END}")
        print(f"\n  Start server with: {Colors.YELLOW}python runner.py{Colors.END}")
        return False
    
    if not account.get('success'):
        print(f"  {Colors.RED}⚠️  Trading not configured{Colors.END}")
        print(f"\n  Add Alpaca API keys to .env file")
        return False
    
    acc = account.get('account', {})
    
    # Portfolio Box
    portfolio_value = acc.get('portfolio_value', 0)
    cash = acc.get('cash', 0)
    equity = acc.get('equity', 0)
    buying_power = acc.get('buying_power', 0)
    
    # Calculate P&L
    starting_balance = 100000
    total_pl = portfolio_value - starting_balance
    pl_pct = (total_pl / starting_balance) * 100 if starting_balance > 0 else 0
    
    portfolio_content = f"""  Portfolio Value: {format_money(portfolio_value)}
  Cash Available:  {Colors.WHITE}${cash:,.2f}{Colors.END}
  Equity:          {Colors.WHITE}${equity:,.2f}{Colors.END}
  Buying Power:    {Colors.WHITE}${buying_power:,.2f}{Colors.END}
  
  Total P&L:       {format_money(total_pl)} ({pl_pct:+.2f}%)
  Status:          {"🟢 TRADING ACTIVE" if pl_pct != 0 else "⏳ WAITING FOR TRADES"}"""
    
    print(f"  {Colors.BOLD}💰 PORTFOLIO SUMMARY{Colors.END}")
    print(f"  ┌{'─' * 68}┐")
    for line in portfolio_content.split('\n'):
        print(f"  │ {line:<66} │")
    print(f"  └{'─' * 68}┘\n")
    
    # Positions Box
    if positions and positions.get('success'):
        pos_list = positions.get('positions', [])
        
        print(f"  {Colors.BOLD}📊 ACTIVE POSITIONS ({len(pos_list)}){Colors.END}")
        print(f"  ┌{'─' * 68}┐")
        print(f"  │ {'Symbol':<8} {'Qty':>8} {'Entry':>12} {'Current':>12} {'P&L':>14} │")
        print(f"  ├{'─' * 68}┤")
        
        if pos_list:
            for pos in pos_list:
                symbol = pos.get('symbol', 'N/A')
                qty = pos.get('qty', 0)
                entry = pos.get('avg_entry_price', 0)
                current = pos.get('current_price', 0)
                pl = pos.get('unrealized_pl', 0)
                pl_pct = pos.get('unrealized_plpc', 0)
                
                pl_str = f"{pl:+.2f} ({pl_pct:+.1f}%)"
                pl_colored = f"{Colors.GREEN if pl >= 0 else Colors.RED}{pl_str:<14}{Colors.END}"
                
                print(f"  │ {symbol:<8} {qty:>8.1f} ${entry:>10.2f} ${current:>10.2f} {pl_colored} │")
        else:
            print(f"  │ {'(No active positions - Strategy waiting for signals)':^66} │")
        
        print(f"  └{'─' * 68}┘\n")
    
    # Recent Orders Box
    if orders and orders.get('success'):
        order_list = orders.get('orders', [])
        recent_orders = order_list[:5]  # Last 5 orders
        
        print(f"  {Colors.BOLD}📝 RECENT ORDERS{Colors.END}")
        print(f"  ┌{'─' * 68}┐")
        print(f"  │ {'Time':<20} {'Symbol':<8} {'Side':<6} {'Qty':>8} {'Status':<12} │")
        print(f"  ├{'─' * 68}┤")
        
        if recent_orders:
            for order in recent_orders:
                created = order.get('created_at', '')
                if created:
                    try:
                        created = created.split('T')[1][:8]  # Extract time
                    except:
                        pass
                symbol = order.get('symbol', 'N/A')
                side = order.get('side', 'N/A').upper()
                qty = order.get('qty', 0)
                status = order.get('status', 'N/A')
                
                side_color = Colors.GREEN if side == 'BUY' else Colors.RED
                
                print(f"  │ {created:<20} {symbol:<8} {side_color}{side:<6}{Colors.END} {qty:>8} {status:<12} │")
        else:
            print(f"  │ {'(No orders yet - Strategy will place orders automatically)':^66} │")
        
        print(f"  └{'─' * 68}┘\n")
    
    # Strategy Status
    print(f"  {Colors.BOLD}🤖 STRATEGY STATUS{Colors.END}")
    print(f"  ┌{'─' * 68}┐")
    print(f"  │ {'Strategy':<20} {'MOMENTUM + RSI (High-Frequency)'}{' ':26} │")
    print(f"  │ {'Target Return':<20} {'50-60% Annually'}{' ':28} │")
    print(f"  │ {'Monitoring':<20} {'NVDA, TSLA, AAPL, MSFT, GOOGL, AMD'}{' ':16} │")
    print(f"  │ {'Check Interval':<20} {'Every 5 minutes'}{' ':35} │")
    print(f"  │ {'Auto-Trading':<20} {'✅ ENABLED' if account else '❌ DISABLED'}{' ':37} │")
    print(f"  └{'─' * 68}┘\n")
    
    # Instructions
    print(f"  {Colors.YELLOW}Commands:{Colors.END}")
    print(f"    • View in browser: {Colors.CYAN}http://localhost:5000{Colors.END}")
    print(f"    • Check account:   {Colors.CYAN}curl http://localhost:5000/api/trading/account{Colors.END}")
    print(f"    • Run strategy:    {Colors.CYAN}curl -X POST http://localhost:5000/api/trading/strategy/execute{Colors.END}")
    print(f"    • Stop server:     {Colors.CYAN}Ctrl+C{Colors.END}")
    print(f"\n  {Colors.GREEN}✓ System running normally. Press Ctrl+C to exit dashboard.{Colors.END}\n")
    
    return True

def main():
    """Main loop"""
    print(f"\n{Colors.BOLD}Starting Alpha Junior Dashboard...{Colors.END}")
    print(f"Connecting to http://localhost:5000...")
    
    try:
        while True:
            success = display_dashboard()
            
            if not success:
                print(f"\n  Retrying in 5 seconds...")
                time.sleep(5)
            else:
                # Refresh every 3 seconds
                time.sleep(3)
                
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Dashboard stopped.{Colors.END}")
        sys.exit(0)

if __name__ == '__main__':
    main()
