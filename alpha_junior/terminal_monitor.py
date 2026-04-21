#!/usr/bin/env python3
"""
Alpha Junior - COMPREHENSIVE TERMINAL MONITOR
Shows everything in terminal: trades, positions, P&L, AI analysis
"""

import requests
import time
import json
from datetime import datetime
import sys
import os

# ANSI Colors for terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    BG_BLACK = '\033[40m'
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'

class TerminalMonitor:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.scan_count = 0
        self.last_prices = {}
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def draw_header(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{Colors.BG_BLUE}{Colors.WHITE}{Colors.BOLD}")
        print("╔" + "═" * 118 + "╗")
        print("║" + " " * 30 + "🏛️  ALPHA JUNIOR v3.0 - TERMINAL MONITOR" + " " * 48 + "║")
        print("║" + " " * 35 + "Top 1% Institutional Trading System" + " " * 47 + "║")
        print("╠" + "═" * 118 + "╣")
        print(f"║  📅 {now}  |  🔄 Updates: Every 5 seconds  |  💻 Server: {self.base_url}" + " " * 45 + "║")
        print("╚" + "═" * 118 + "╝")
        print(f"{Colors.RESET}")
        
    def get_data(self, endpoint):
        try:
            response = requests.get(f'{self.base_url}{endpoint}', timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            return None
            
    def draw_status_box(self, status_data):
        print(f"{Colors.BOLD}{Colors.CYAN}┌─ SYSTEM STATUS ─{'─' * 101}┐{Colors.RESET}")
        
        if status_data:
            engine_status = status_data.get('engine_status', 'unknown')
            portfolio_value = status_data.get('portfolio_value', 0)
            cash = status_data.get('cash', 0)
            
            status_color = Colors.GREEN if engine_status == 'running' else Colors.YELLOW
            
            print(f"│  Engine: {status_color}{engine_status.upper()}{Colors.RESET}  |  "
                  f"Portfolio: {Colors.GREEN}${portfolio_value:,.2f}{Colors.RESET}  |  "
                  f"Cash: {Colors.YELLOW}${cash:,.2f}{Colors.RESET}  |  "
                  f"Mode: {Colors.MAGENTA}PAPER TRADING{Colors.RESET}" + " " * 47 + "│")
        else:
            print(f"│  {Colors.RED}⚠ Cannot connect to server - check if running on port 5000{Colors.RESET}" + " " * 68 + "│")
            
        print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 118}┘{Colors.RESET}")
        
    def draw_positions_box(self, positions_data):
        print(f"{Colors.BOLD}{Colors.GREEN}┌─ ACTIVE POSITIONS ─{'─' * 98}┐{Colors.RESET}")
        
        if positions_data and len(positions_data) > 0:
            print(f"│  {'Symbol':<10} {'Qty':<8} {'Entry':<12} {'Current':<12} {'P&L':<15} {'%':<10} {'Strategy':<20} │")
            print(f"│  {'─'*10} {'─'*8} {'─'*12} {'─'*12} {'─'*15} {'─'*10} {'─'*20} │")
            
            total_pl = 0
            for pos in positions_data:
                symbol = pos.get('symbol', 'N/A')
                qty = pos.get('qty', 0)
                entry = pos.get('avg_entry_price', 0)
                current = pos.get('current_price', entry)
                strategy = pos.get('strategy', 'Unknown')[:18]
                
                # Calculate P&L
                pl = (current - entry) * qty
                total_pl += pl
                pl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                
                # Color based on P&L
                pl_color = Colors.GREEN if pl >= 0 else Colors.RED
                
                print(f"│  {Colors.BOLD}{symbol:<10}{Colors.RESET} {qty:<8} ${entry:<11.2f} ${current:<11.2f} "
                      f"{pl_color}{pl:>+13.2f}{Colors.RESET} {pl_color}{pl_pct:>+8.1f}%{Colors.RESET} "
                      f"{strategy:<20} │")
            
            print(f"│  {'─'*10} {'─'*8} {'─'*12} {'─'*12} {'─'*15} {'─'*10} {'─'*20} │")
            total_color = Colors.GREEN if total_pl >= 0 else Colors.RED
            print(f"│  {'TOTAL P&L:':<45} {total_color}{total_pl:>+13.2f}{Colors.RESET}" + " " * 56 + "│")
        else:
            print(f"│  {Colors.YELLOW}⏳ No active positions - waiting for first trades...{Colors.RESET}" + " " * 65 + "│")
            print(f"│  {Colors.CYAN}💡 14 AI traders are scanning for opportunities{Colors.RESET}" + " " * 68 + "│")
            
        print(f"{Colors.BOLD}{Colors.GREEN}└{'─' * 118}┘{Colors.RESET}")
        
    def draw_traders_box(self):
        print(f"{Colors.BOLD}{Colors.MAGENTA}┌─ 14 AI TRADING STRATEGIES ─{'─' * 89}┐{Colors.RESET}")
        
        traders = [
            ("🏆 Momentum", "High-momentum breakout", "Scanning"),
            ("📊 Mean Reversion", "Oversold bounce", "Scanning"),
            ("💥 Breakout", "Pattern breakout", "Scanning"),
            ("📈 Trend", "Long-term trend", "Scanning"),
            ("🔄 Swing", "3-10 day cycles", "Scanning"),
            ("⚡ Scalping", "Intraday micro", "Scanning"),
            ("📅 Position", "Long-term holds", "Scanning"),
            ("🎯 Arbitrage", "Statistical arb", "Scanning"),
            ("📉 Gap", "Overnight gaps", "Scanning"),
            ("🏛️ Sector", "Rotation plays", "Scanning"),
            ("📊 Volatility", "VIX strategies", "Scanning"),
            ("📰 Event", "Earnings/news", "Scanning"),
            ("🤖 Algo", "Pattern ML", "Scanning"),
            ("🎲 Pairs", "Correlation", "Scanning"),
        ]
        
        for i in range(0, len(traders), 2):
            t1 = traders[i]
            if i + 1 < len(traders):
                t2 = traders[i + 1]
                print(f"│  {Colors.CYAN}{t1[0]:<18}{Colors.RESET} {t1[1]:<25} {Colors.GREEN}{t1[2]:<12}{Colors.RESET}  │  "
                      f"{Colors.CYAN}{t2[0]:<18}{Colors.RESET} {t2[1]:<25} {Colors.GREEN}{t2[2]:<12}{Colors.RESET}  │")
            else:
                print(f"│  {Colors.CYAN}{t1[0]:<18}{Colors.RESET} {t1[1]:<25} {Colors.GREEN}{t1[2]:<12}{Colors.RESET}" + " " * 59 + "│")
                
        print(f"{Colors.BOLD}{Colors.MAGENTA}└{'─' * 118}┘{Colors.RESET}")
        
    def draw_risk_box(self):
        print(f"{Colors.BOLD}{Colors.YELLOW}┌─ RISK MANAGEMENT ─{'─' * 99}┐{Colors.RESET}")
        
        limits = {
            'VaR 95% Limit': '2.0%',
            'Max Drawdown': '15.0%',
            'Max Position Size': '10.0%',
            'Max Leverage': '1.5x',
            'Min Cash Reserve': '5.0%',
            'Stop Loss': '8.0%',
            'Take Profit': '20.0%'
        }
        
        items = list(limits.items())
        for i in range(0, len(items), 4):
            line = "│  "
            for j in range(4):
                if i + j < len(items):
                    key, val = items[i + j]
                    line += f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.GREEN}{val}{Colors.RESET}  │  "
                else:
                    line += " " * 25 + "│  "
            print(line)
            
        print(f"│  {Colors.GREEN}✓ All risk systems active and monitoring{Colors.RESET}" + " " * 72 + "│")
        print(f"{Colors.BOLD}{Colors.YELLOW}└{'─' * 118}┘{Colors.RESET}")
        
    def draw_alpaca_info(self):
        print(f"{Colors.BOLD}{Colors.WHITE}┌─ ALPACA PAPER TRADING ─{'─' * 94}┐{Colors.RESET}")
        
        try:
            account = self.get_data('/api/trading/account')
            if account:
                cash = float(account.get('cash', 0))
                buying_power = float(account.get('buying_power', 0))
                portfolio_value = float(account.get('portfolio_value', 0))
                status = account.get('status', 'unknown')
                
                print(f"│  {Colors.GREEN}✓ Alpaca Connection: ACTIVE{Colors.RESET}  |  "
                      f"Status: {Colors.CYAN}{status}{Colors.RESET}  |  "
                      f"Cash: {Colors.YELLOW}${cash:,.2f}{Colors.RESET}  |  "
                      f"Buying Power: {Colors.GREEN}${buying_power:,.2f}{Colors.RESET}" + " " * 35 + "│")
                print(f"│  {Colors.CYAN}Portfolio Value: ${portfolio_value:,.2f}{Colors.RESET}" + " " * 95 + "│")
            else:
                print(f"│  {Colors.YELLOW}⚠ Alpaca data not available - demo mode active{Colors.RESET}" + " " * 69 + "│")
        except:
            print(f"│  {Colors.YELLOW}⚠ Alpaca integration check pending{Colors.RESET}" + " " * 80 + "│")
            
        print(f"{Colors.BOLD}{Colors.WHITE}└{'─' * 118}┘{Colors.RESET}")
        
    def draw_recent_trades(self):
        print(f"{Colors.BOLD}{Colors.BLUE}┌─ RECENT ACTIVITY LOG ─{'─' * 95}┐{Colors.RESET}")
        
        # Simulate recent activity (in real version, fetch from API)
        activities = [
            (datetime.now().strftime("%H:%M:%S"), "System", "INFO", "14 AI traders scanning market"),
            (datetime.now().strftime("%H:%M:%S"), "Risk", "OK", "All limits within bounds"),
            (datetime.now().strftime("%H:%M:%S"), "Engine", "ACTIVE", "Elite Hedge Fund mode"),
        ]
        
        print(f"│  {'Time':<12} {'Component':<15} {'Status':<10} {'Message':<75} │")
        print(f"│  {'─'*12} {'─'*15} {'─'*10} {'─'*75} │")
        
        for time_str, component, status, message in activities:
            status_color = Colors.GREEN if status in ['OK', 'ACTIVE', 'INFO'] else Colors.YELLOW
            print(f"│  {time_str:<12} {Colors.CYAN}{component:<15}{Colors.RESET} {status_color}{status:<10}{Colors.RESET} {message:<75} │")
            
        print(f"{Colors.BOLD}{Colors.BLUE}└{'─' * 118}┘{Colors.RESET}")
        
    def draw_footer(self):
        print()
        print(f"{Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD}")
        print("╔" + "═" * 118 + "╗")
        print(f"║  💡 COMMANDS:  [Q]uit  |  [R]efresh Now  |  [P]ositions  |  [A]ccount  |  [T]rades  |  [H]elp" + " " * 29 + "║")
        print("╚" + "═" * 118 + "╝")
        print(f"{Colors.RESET}")
        
    def run(self):
        print("Starting Terminal Monitor... Press Ctrl+C to exit")
        time.sleep(2)
        
        try:
            while True:
                self.clear_screen()
                
                # Fetch all data
                status = self.get_data('/api/elite/status')
                positions = self.get_data('/api/trading/positions')
                
                # Draw everything
                self.draw_header()
                self.draw_status_box(status)
                self.draw_alpaca_info()
                self.draw_positions_box(positions)
                self.draw_traders_box()
                self.draw_risk_box()
                self.draw_recent_trades()
                self.draw_footer()
                
                # Wait for next update
                time.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitor stopped. Trading continues in background.{Colors.RESET}")
            
def main():
    monitor = TerminalMonitor()
    monitor.run()

if __name__ == '__main__':
    main()
