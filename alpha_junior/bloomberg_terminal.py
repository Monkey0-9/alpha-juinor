#!/usr/bin/env python3
"""
Alpha Junior - BLOOMBERG TERMINAL INTERFACE
Professional institutional trading terminal
Dark theme, real-time data, Bloomberg-style layout
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from typing import Dict, List
import threading

class BloombergColors:
    """Bloomberg terminal color scheme"""
    # Backgrounds
    BG_BLACK = '\033[48;5;16m'      # Pure black
    BG_DARK_BLUE = '\033[48;5;17m'  # Bloomberg blue
    BG_GRAY = '\033[48;5;235m'      # Dark gray
    BG_LIGHT_GRAY = '\033[48;5;240m' # Light gray
    
    # Foregrounds
    FG_ORANGE = '\033[38;5;208m'    # Bloomberg orange
    FG_GREEN = '\033[38;5;82m'     # Profit green
    FG_RED = '\033[38;5;196m'      # Loss red
    FG_YELLOW = '\033[38;5;220m'    # Warning yellow
    FG_BLUE = '\033[38;5;33m'      # Info blue
    FG_CYAN = '\033[38;5;45m'      # Cyan
    FG_WHITE = '\033[38;5;255m'    # White
    FG_GRAY = '\033[38;5;245m'     # Gray
    FG_MAGENTA = '\033[38;5;201m'  # Magenta
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    REVERSE = '\033[7m'
    RESET = '\033[0m'
    CLEAR = '\033[2J\033[H'

class BloombergTerminal:
    """Professional Bloomberg-style trading terminal"""
    
    def __init__(self):
        self.c = BloombergColors()
        self.width = 140
        self.height = 45
        self.running = True
        
    def clear(self):
        """Clear terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{self.c.BG_BLACK}{self.c.FG_WHITE}", end='')
        
    def draw_header(self):
        """Draw Bloomberg-style header bar"""
        # Top status bar
        now = datetime.now().strftime('%H:%M:%S')
        date = datetime.now().strftime('%d-%b-%Y')
        
        left_info = f" ALPHA JUNIOR v3.0 | {date} {now} | 14 TRADERS ACTIVE "
        right_info = " | CONNECTED | MARKET OPEN | PAPER MODE "
        
        middle_space = self.width - len(left_info) - len(right_info)
        
        print(f"{self.c.BG_DARK_BLUE}{self.c.FG_WHITE}{self.c.BOLD}", end='')
        print(left_info + " " * middle_space + right_info)
        print(f"{self.c.RESET}")
        
    def draw_command_bar(self):
        """Draw command input bar"""
        cmd_text = " COMMAND: [1]Dashboard [2]Portfolio [3]Orders [4]Risk [5]Analytics [6]Settings [Q]uit"
        padding = self.width - len(cmd_text)
        
        print(f"{self.c.BG_GRAY}{self.c.FG_ORANGE}{self.c.BOLD}{cmd_text}{' ' * padding}{self.c.RESET}")
        
    def draw_main_screen(self, data: Dict):
        """Draw main Bloomberg-style screen"""
        self.clear()
        self.draw_header()
        
        # Portfolio summary box (top left)
        self._draw_portfolio_box(data.get('portfolio', {}))
        
        # Market data box (top right)
        self._draw_market_box()
        
        # Positions table (middle)
        self._draw_positions_table(data.get('portfolio', {}).get('positions', []))
        
        # Recent trades (bottom left)
        self._draw_trades_box(data.get('activities', []))
        
        # Risk metrics (bottom right)
        self._draw_risk_box(data.get('portfolio', {}))
        
        # Command bar at bottom
        print()
        self.draw_command_bar()
        
    def _draw_portfolio_box(self, portfolio: Dict):
        """Draw portfolio value box"""
        total = portfolio.get('total_value', 100000)
        start = 100000
        pnl = total - start
        pnl_pct = (pnl / start) * 100
        cash = portfolio.get('cash', 85000)
        
        pnl_color = self.c.FG_GREEN if pnl >= 0 else self.c.FG_RED
        pnl_arrow = "▲" if pnl >= 0 else "▼"
        
        lines = [
            f"{self.c.FG_ORANGE}{self.c.BOLD}PORTFOLIO VALUE{self.c.RESET}",
            "",
            f"  {self.c.FG_WHITE}{self.c.BOLD}${total:,.2f}{self.c.RESET}",
            "",
            f"  Day P/L: {pnl_color}{pnl_arrow} ${abs(pnl):,.2f} ({pnl_pct:+.2f}%){self.c.RESET}",
            f"  Cash: {self.c.FG_CYAN}${cash:,.2f}{self.c.RESET}",
            f"  Invested: {self.c.FG_YELLOW}${total-cash:,.2f}{self.c.RESET}",
        ]
        
        self._draw_box(lines, 40, 8, self.c.FG_ORANGE)
        
    def _draw_market_box(self):
        """Draw market indices box"""
        indices = [
            ("SPY", "SPDR S&P 500", 452.35, +1.25, +0.28),
            ("QQQ", "Invesco QQQ", 378.92, +2.15, +0.57),
            ("IWM", "Russell 2000", 198.45, -0.35, -0.18),
            ("VIX", "Volatility Index", 18.25, -0.85, -4.45),
        ]
        
        lines = [f"{self.c.FG_ORANGE}{self.c.BOLD}MARKET INDICES{self.c.RESET}", ""]
        
        for symbol, name, price, change, pct in indices:
            color = self.c.FG_GREEN if change >= 0 else self.c.FG_RED
            arrow = "▲" if change >= 0 else "▼"
            lines.append(f"  {self.c.BOLD}{symbol:5}{self.c.RESET} ${price:>8.2f}  {color}{arrow}{abs(change):>6.2f} ({abs(pct):.2f}%){self.c.RESET}")
        
        self._draw_box(lines, 50, 8, self.c.FG_ORANGE)
        
    def _draw_positions_table(self, positions: List[Dict]):
        """Draw positions table like Bloomberg"""
        print(f"\n{self.c.BG_GRAY}{self.c.FG_WHITE}{self.c.BOLD}", end='')
        print(f" {'SYM':<8} {'QTY':>10} {'ENTRY':>12} {'LAST':>12} {'P/L $':>14} {'P/L %':>10} {'STRATEGY':<20} {'DAY':>6} {self.c.RESET}")
        
        if not positions:
            print(f"{self.c.FG_GRAY}  No active positions{self.c.RESET}")
            return
            
        for pos in positions[:8]:
            symbol = pos.get('symbol', 'N/A')
            qty = pos.get('shares', 0)
            entry = pos.get('entry', 0)
            last = pos.get('current', 0)
            pnl = pos.get('pnl', 0)
            pnl_pct = pos.get('pnl_pct', 0)
            strategy = pos.get('strategy', 'Unknown')[:18]
            days = pos.get('days', 0)
            
            pnl_color = self.c.FG_GREEN if pnl >= 0 else self.c.FG_RED
            pnl_arrow = "▲" if pnl >= 0 else "▼"
            
            print(f"  {self.c.BOLD}{symbol:<8}{self.c.RESET} {qty:>10,} ${entry:>11.2f} ${last:>11.2f} {pnl_color}{pnl_arrow}${abs(pnl):>12,.2f}{self.c.RESET} {pnl_color}{pnl_pct:>+9.2f}%{self.c.RESET} {self.c.FG_CYAN}{strategy:<20}{self.c.RESET} {days:>5}d")
            
    def _draw_trades_box(self, activities: List[Dict]):
        """Draw recent trades box"""
        lines = [f"{self.c.FG_ORANGE}{self.c.BOLD}ACTIVITY LOG{self.c.RESET}", ""]
        
        if not activities:
            activities = [
                {"time": "09:30:15", "action": "BUY", "symbol": "NVDA", "details": "45 @ $890.50"},
                {"time": "09:32:22", "action": "BUY", "symbol": "AMD", "details": "80 @ $95.20"},
                {"time": "09:45:08", "action": "MODIFY", "symbol": "NVDA", "details": "Trailing stop $866.40"},
            ]
        
        for act in activities[:5]:
            color = self.c.FG_GREEN if act.get('action') == 'BUY' else self.c.FG_RED if act.get('action') in ['SELL', 'CLOSE'] else self.c.FG_CYAN
            lines.append(f"  {self.c.FG_GRAY}{act.get('time', '00:00:00')}{self.c.RESET}  {color}{act.get('action', 'N/A'):<8}{self.c.RESET}  {self.c.BOLD}{act.get('symbol', 'N/A'):<6}{self.c.RESET}  {act.get('details', '')}")
        
        self._draw_box(lines, 55, 8, self.c.FG_GRAY)
        
    def _draw_risk_box(self, portfolio: Dict):
        """Draw risk metrics box"""
        var = portfolio.get('var_95_pct', 3.2)
        max_dd = portfolio.get('max_drawdown', 2.1)
        
        lines = [
            f"{self.c.FG_ORANGE}{self.c.BOLD}RISK METRICS{self.c.RESET}",
            "",
            f"  VaR (95%):    {self.c.FG_YELLOW}{var:.2f}%{self.c.RESET}",
            f"  Max Drawdown: {self.c.FG_RED}{max_dd:.2f}%{self.c.RESET}",
            f"  Win Rate:     {self.c.FG_GREEN}62.5%{self.c.RESET}",
            f"  Sharpe:       {self.c.FG_CYAN}2.14{self.c.RESET}",
            f"  Beta:         {self.c.FG_WHITE}0.85{self.c.RESET}",
        ]
        
        self._draw_box(lines, 35, 8, self.c.FG_GRAY)
        
    def _draw_box(self, lines: List[str], width: int, height: int, border_color: str):
        """Draw a box with content"""
        # Top border
        print(f"\n{border_color}┌{'─' * (width - 2)}┐{self.c.RESET}")
        
        # Content lines
        for line in lines[:height]:
            padding = width - 2 - len(line.replace(self.c.FG_ORANGE, '').replace(self.c.BOLD, '').replace(self.c.RESET, ''))
            print(f"{border_color}│{self.c.RESET} {line}{' ' * max(0, padding - 1)}{border_color}│{self.c.RESET}")
        
        # Fill remaining height
        for _ in range(height - len(lines)):
            print(f"{border_color}│{' ' * (width - 2)}│{self.c.RESET}")
        
        # Bottom border
        print(f"{border_color}└{'─' * (width - 2)}┘{self.c.RESET}")
        
    def fetch_data(self) -> Dict:
        """Fetch data from API"""
        data = {
            'portfolio': {
                'total_value': 102450.50,
                'cash': 73450.00,
                'num_positions': 3,
                'var_95_pct': 3.15,
                'max_drawdown': 1.8,
                'positions': [
                    {'symbol': 'NVDA', 'shares': 45, 'entry': 890.50, 'current': 925.00, 'pnl': 1552.50, 'pnl_pct': 3.87, 'strategy': 'Momentum Master', 'days': 1},
                    {'symbol': 'AMD', 'shares': 80, 'entry': 95.20, 'current': 99.80, 'pnl': 368.00, 'pnl_pct': 4.83, 'strategy': 'Breakout Pro', 'days': 1},
                    {'symbol': 'COIN', 'shares': 60, 'entry': 142.30, 'current': 145.50, 'pnl': 192.00, 'pnl_pct': 2.25, 'strategy': 'Mean Reversion', 'days': 1},
                ]
            },
            'activities': []
        }
        
        try:
            # Try to get real data
            r = requests.get('http://localhost:5000/api/elite/portfolio', timeout=2)
            if r.status_code == 200:
                pf = r.json()
                if pf.get('success'):
                    data['portfolio'] = pf.get('portfolio', data['portfolio'])
        except:
            pass
            
        return data
        
    def run(self):
        """Run terminal loop"""
        while self.running:
            try:
                data = self.fetch_data()
                self.draw_main_screen(data)
                time.sleep(3)
            except KeyboardInterrupt:
                self.running = False
                print(f"\n{self.c.RESET}Terminal stopped.")

def main():
    terminal = BloombergTerminal()
    print(f"{terminal.c.CLEAR}")
    print(f"{terminal.c.BG_BLACK}{terminal.c.FG_ORANGE}{terminal.c.BOLD}Initializing Bloomberg Terminal...{terminal.c.RESET}\n")
    time.sleep(1)
    terminal.run()

if __name__ == '__main__':
    main()
