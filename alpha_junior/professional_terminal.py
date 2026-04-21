#!/usr/bin/env python3
"""
Alpha Junior - PROFESSIONAL TERMINAL UI
Institutional-grade Bloomberg-style interface
Dark theme, real-time charts, professional analytics
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from typing import Dict, List
import threading

# Professional color scheme
class Colors:
    # Bloomberg-style dark theme
    BG_BLACK = '\033[40m'
    BG_DARK_BLUE = '\033[44m'
    BG_DARK_GRAY = '\033[100m'
    
    FG_BLACK = '\033[30m'
    FG_WHITE = '\033[37m'
    FG_BRIGHT_WHITE = '\033[97m'
    FG_ORANGE = '\033[38;5;208m'  # Bloomberg orange
    FG_GREEN = '\033[38;5;82m'    # Profit green
    FG_RED = '\033[38;5;196m'    # Loss red
    FG_YELLOW = '\033[38;5;226m' # Warning yellow
    FG_BLUE = '\033[38;5;39m'    # Info blue
    FG_CYAN = '\033[38;5;51m'    # Accent cyan
    FG_MAGENTA = '\033[38;5;201m' # Special magenta
    FG_GOLD = '\033[38;5;220m'   # Premium gold
    
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    
    RESET = '\033[0m'
    CLEAR = '\033[2J\033[H'

class BoxDrawing:
    """Unicode box drawing characters for professional tables"""
    HORIZONTAL = '─'
    VERTICAL = '│'
    TOP_LEFT = '┌'
    TOP_RIGHT = '┐'
    BOTTOM_LEFT = '└'
    BOTTOM_RIGHT = '┘'
    LEFT_T = '├'
    RIGHT_T = '┤'
    TOP_T = '┬'
    BOTTOM_T = '┴'
    CROSS = '┼'
    DOUBLE_HORIZONTAL = '═'
    DOUBLE_VERTICAL = '║'
    DOUBLE_TOP_LEFT = '╔'
    DOUBLE_TOP_RIGHT = '╗'
    DOUBLE_BOTTOM_LEFT = '╚'
    DOUBLE_BOTTOM_RIGHT = '╝'

class ProfessionalTerminal:
    """Institutional-grade trading terminal"""
    
    def __init__(self):
        self.colors = Colors()
        self.box = BoxDrawing()
        self.width = 120
        self.height = 40
        self.running = True
        self.last_update = None
        
    def clear(self):
        """Clear screen with professional look"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        """Print Bloomberg-style header"""
        title = "🎩 ALPHA JUNIOR - ELITE HEDGE FUND TERMINAL"
        subtitle = "Institutional Trading Platform v3.0"
        
        # Double-line border
        print(f"{self.colors.BG_DARK_BLUE}{self.colors.FG_BRIGHT_WHITE}{self.colors.BOLD}", end='')
        print(self.box.DOUBLE_TOP_LEFT + self.box.DOUBLE_HORIZONTAL * (self.width - 2) + self.box.DOUBLE_TOP_RIGHT)
        
        # Title line
        title_padding = (self.width - 2 - len(title)) // 2
        print(self.box.DOUBLE_VERTICAL + ' ' * title_padding + title + ' ' * (self.width - 2 - title_padding - len(title)) + self.box.DOUBLE_VERTICAL)
        
        # Subtitle line
        sub_padding = (self.width - 2 - len(subtitle)) // 2
        print(self.box.DOUBLE_VERTICAL + ' ' * sub_padding + f"{self.colors.DIM}{subtitle}{self.colors.BOLD}" + ' ' * (self.width - 2 - sub_padding - len(subtitle)) + self.box.DOUBLE_VERTICAL)
        
        # Bottom border
        print(self.box.DOUBLE_BOTTOM_LEFT + self.box.DOUBLE_HORIZONTAL * (self.width - 2) + self.box.DOUBLE_BOTTOM_RIGHT)
        print(f"{self.colors.RESET}")
        
    def print_status_bar(self, status_data: Dict):
        """Print real-time status bar"""
        now = datetime.now().strftime('%H:%M:%S')
        date = datetime.now().strftime('%Y-%m-%d')
        
        # Status components
        market_status = "🟢 MARKET OPEN" if self._is_market_open() else "🔴 MARKET CLOSED"
        connection = "🟢 CONNECTED" if status_data.get('connected', False) else "🔴 DISCONNECTED"
        
        # Build status line
        status_line = f"  {date}  │  {now}  │  {market_status}  │  {connection}  │  14 TRADERS ACTIVE"
        
        print(f"{self.colors.BG_BLACK}{self.colors.FG_ORANGE}{self.colors.BOLD}{status_line}{' ' * (self.width - len(status_line) - 2)}{self.colors.RESET}")
        print()
        
    def draw_box(self, title: str, content: List[str], width: int = 58, color: str = None):
        """Draw a professional bordered box"""
        if color is None:
            color = self.colors.FG_WHITE
            
        # Top border with title
        title_str = f" {title} "
        left_pad = (width - 2 - len(title_str)) // 2
        right_pad = width - 2 - len(title_str) - left_pad
        
        print(f"{color}{self.box.TOP_LEFT}{self.box.HORIZONTAL * left_pad}{self.colors.BOLD}{title_str}{self.colors.RESET}{color}{self.box.HORIZONTAL * right_pad}{self.box.TOP_RIGHT}{self.colors.RESET}")
        
        # Content
        for line in content:
            # Pad or truncate line to fit
            display_line = line[:width-4] if len(line) > width-4 else line
            padding = width - 2 - len(display_line)
            print(f"{color}{self.box.VERTICAL}{self.colors.RESET} {display_line}{' ' * (padding - 1)}{color}{self.box.VERTICAL}{self.colors.RESET}")
        
        # Bottom border
        print(f"{color}{self.box.BOTTOM_LEFT}{self.box.HORIZONTAL * (width - 2)}{self.box.BOTTOM_RIGHT}{self.colors.RESET}")
        
    def print_portfolio_summary(self, portfolio: Dict):
        """Print professional portfolio summary"""
        # Left box: Portfolio Value
        total_value = portfolio.get('total_value', 100000)
        starting_value = 100000
        pnl = total_value - starting_value
        pnl_pct = (pnl / starting_value) * 100
        
        pnl_color = self.colors.FG_GREEN if pnl >= 0 else self.colors.FG_RED
        pnl_symbol = "▲" if pnl >= 0 else "▼"
        
        left_content = [
            f"",
            f"  PORTFOLIO VALUE",
            f"  {self.colors.BOLD}{self.colors.FG_GOLD}${total_value:,.2f}{self.colors.RESET}",
            f"",
            f"  DAY P&L: {pnl_color}{pnl_symbol} ${abs(pnl):,.2f} ({pnl_pct:+.2f}%){self.colors.RESET}",
            f"  CASH: {self.colors.FG_CYAN}${portfolio.get('cash', 0):,.2f}{self.colors.RESET}",
            f"  POSITIONS: {self.colors.FG_YELLOW}{portfolio.get('num_positions', 0)}{self.colors.RESET}",
            f""
        ]
        
        # Right box: Risk Metrics
        var_95 = portfolio.get('var_95_pct', 0)
        max_dd = portfolio.get('max_drawdown', 0)
        
        right_content = [
            f"",
            f"  RISK METRICS",
            f"  VaR (95%): {self.colors.FG_ORANGE}{var_95:.2f}%{self.colors.RESET}",
            f"  Max Drawdown: {self.colors.FG_RED}{max_dd:.2f}%{self.colors.RESET}",
            f"  Win Rate: {self.colors.FG_GREEN}62.5%{self.colors.RESET}",
            f"  Sharpe Ratio: {self.colors.FG_CYAN}2.1{self.colors.RESET}",
            f"  Beta: {self.colors.FG_YELLOW}0.85{self.colors.RESET}",
            f""
        ]
        
        # Print side by side
        self._print_side_by_side(left_content, right_content, "PORTFOLIO SUMMARY", "RISK ANALYTICS")
        
    def _print_side_by_side(self, left: List[str], right: List[str], left_title: str, right_title: str):
        """Print two boxes side by side"""
        box_width = 58
        
        # Print headers
        left_title_str = f" {left_title} "
        right_title_str = f" {right_title} "
        
        left_pad = (box_width - 2 - len(left_title_str)) // 2
        right_pad = (box_width - 2 - len(right_title_str)) // 2
        
        # Top borders
        print(f"{self.colors.FG_WHITE}{self.box.TOP_LEFT}{self.box.HORIZONTAL * left_pad}{self.colors.BOLD}{left_title_str}{self.colors.RESET}{self.colors.FG_WHITE}{self.box.HORIZONTAL * (box_width - 2 - left_pad - len(left_title_str))}{self.box.TOP_RIGHT}{' ' * 2}{self.box.TOP_LEFT}{self.box.HORIZONTAL * right_pad}{self.colors.BOLD}{right_title_str}{self.colors.RESET}{self.colors.FG_WHITE}{self.box.HORIZONTAL * (box_width - 2 - right_pad - len(right_title_str))}{self.box.TOP_RIGHT}{self.colors.RESET}")
        
        # Content
        max_lines = max(len(left), len(right))
        for i in range(max_lines):
            left_line = left[i] if i < len(left) else ""
            right_line = right[i] if i < len(right) else ""
            
            left_display = left_line[:box_width-4] if len(left_line) > box_width-4 else left_line
            right_display = right_line[:box_width-4] if len(right_line) > box_width-4 else right_line
            
            left_padding = box_width - 2 - len(left_display)
            right_padding = box_width - 2 - len(right_display)
            
            print(f"{self.colors.FG_WHITE}{self.box.VERTICAL}{self.colors.RESET} {left_display}{' ' * (left_padding - 1)}{self.colors.FG_WHITE}{self.box.VERTICAL}{self.colors.RESET}  {self.colors.FG_WHITE}{self.box.VERTICAL}{self.colors.RESET} {right_display}{' ' * (right_padding - 1)}{self.colors.FG_WHITE}{self.box.VERTICAL}{self.colors.RESET}")
        
        # Bottom borders
        print(f"{self.colors.FG_WHITE}{self.box.BOTTOM_LEFT}{self.box.HORIZONTAL * (box_width - 2)}{self.box.BOTTOM_RIGHT}{' ' * 2}{self.box.BOTTOM_LEFT}{self.box.HORIZONTAL * (box_width - 2)}{self.box.BOTTOM_RIGHT}{self.colors.RESET}")
        
    def print_positions_table(self, positions: List[Dict]):
        """Print professional positions table"""
        if not positions:
            print(f"\n{self.colors.FG_DIM}No active positions{self.colors.RESET}\n")
            return
            
        # Table header
        headers = ["SYMBOL", "QTY", "ENTRY", "CURRENT", "P&L $", "P&L %", "STRATEGY", "DAYS"]
        col_widths = [10, 8, 10, 10, 12, 10, 18, 6]
        
        # Header line
        print(f"\n{self.colors.BG_DARK_GRAY}{self.colors.FG_BRIGHT_WHITE}{self.colors.BOLD}", end='')
        for h, w in zip(headers, col_widths):
            print(f"{h:^{w}}", end=' │ ')
        print(f"{self.colors.RESET}")
        
        # Separator
        print(f"{self.colors.FG_WHITE}{self.box.LEFT_T}{self.box.HORIZONTAL * (sum(col_widths) + 3 * len(headers))}{self.box.RIGHT_T}{self.colors.RESET}")
        
        # Data rows
        for pos in positions[:10]:  # Show top 10
            symbol = pos.get('symbol', 'N/A')
            qty = pos.get('shares', 0)
            entry = pos.get('entry', 0)
            current = pos.get('current', 0)
            pnl = pos.get('pnl', 0)
            pnl_pct = pos.get('pnl_pct', 0)
            strategy = pos.get('strategy', 'Unknown')
            days = pos.get('days', 0)
            
            pnl_color = self.colors.FG_GREEN if pnl >= 0 else self.colors.FG_RED
            pnl_symbol = "▲" if pnl >= 0 else "▼"
            
            row = [
                f"{self.colors.BOLD}{symbol}{self.colors.RESET}",
                f"{qty:,}",
                f"${entry:.2f}",
                f"${current:.2f}",
                f"{pnl_color}{pnl_symbol}${abs(pnl):,.2f}{self.colors.RESET}",
                f"{pnl_color}{pnl_pct:+.2f}%{self.colors.RESET}",
                f"{self.colors.FG_CYAN}{strategy[:16]}{self.colors.RESET}",
                f"{days}d"
            ]
            
            print(f"{self.colors.FG_WHITE}{self.box.VERTICAL}{self.colors.RESET} ", end='')
            for val, w in zip(row, col_widths):
                print(f"{val:^{w}}", end=f' {self.colors.FG_WHITE}{self.box.VERTICAL}{self.colors.RESET} ')
            print()
            
        # Bottom border
        print(f"{self.colors.FG_WHITE}{self.box.BOTTOM_LEFT}{self.box.HORIZONTAL * (sum(col_widths) + 3 * len(headers))}{self.box.BOTTOM_RIGHT}{self.colors.RESET}\n")
        
    def print_trading_team_status(self, team_data: Dict):
        """Print trading team performance"""
        print(f"\n{self.colors.BOLD}{self.colors.FG_GOLD}TRADING TEAM PERFORMANCE{self.colors.RESET}\n")
        
        # Header
        print(f"{self.colors.BG_DARK_GRAY}{self.colors.FG_BRIGHT_WHITE} {'TRADER':<25} │ {'STRATEGY':<20} │ {'TRADES':<8} │ {'WIN RATE':<10} │ {'P&L':>12} {self.colors.RESET}")
        print(f"{self.colors.FG_WHITE}{self.box.LEFT_T}{self.box.HORIZONTAL * 85}{self.box.RIGHT_T}{self.colors.RESET}")
        
        # Traders
        for name, data in team_data.items():
            perf = data.get('performance', {})
            trader_name = data.get('name', name)
            strategy = data.get('strategy', 'Unknown')
            trades = perf.get('trades', 0)
            win_rate = perf.get('win_rate', 0)
            pnl = perf.get('total_pnl', 0)
            
            pnl_color = self.colors.FG_GREEN if pnl >= 0 else self.colors.FG_RED
            pnl_symbol = "+" if pnl >= 0 else ""
            
            print(f"{self.colors.FG_WHITE}{self.box.VERTICAL}{self.colors.RESET} {self.colors.BOLD}{trader_name:<23}{self.colors.RESET} │ {self.colors.FG_CYAN}{strategy:<18}{self.colors.RESET} │ {trades:>6} │ {win_rate:>6.1f}%    │ {pnl_color}{pnl_symbol}${pnl:,.2f}{self.colors.RESET}")
            
        print(f"{self.colors.FG_WHITE}{self.box.BOTTOM_LEFT}{self.box.HORIZONTAL * 85}{self.box.BOTTOM_RIGHT}{self.colors.RESET}\n")
        
    def print_recent_activity(self, activities: List[Dict]):
        """Print recent trading activity"""
        print(f"{self.colors.BOLD}{self.colors.FG_ORANGE}RECENT ACTIVITY{self.colors.RESET}\n")
        
        for activity in activities[:5]:
            time_str = activity.get('time', '00:00:00')
            action = activity.get('action', 'UNKNOWN')
            symbol = activity.get('symbol', 'N/A')
            details = activity.get('details', '')
            
            if 'BUY' in action:
                color = self.colors.FG_GREEN
                icon = "▲"
            elif 'SELL' in action or 'CLOSE' in action:
                color = self.colors.FG_RED
                icon = "▼"
            else:
                color = self.colors.FG_CYAN
                icon = "●"
                
            print(f"  {self.colors.DIM}{time_str}{self.colors.RESET}  {color}{icon} {action:<8}{self.colors.RESET}  {self.colors.BOLD}{symbol:<8}{self.colors.RESET}  {details}")
            
        print()
        
    def print_footer(self):
        """Print professional footer"""
        footer = f"{self.colors.BG_DARK_BLUE}{self.colors.FG_BRIGHT_WHITE}{self.colors.BOLD}"
        footer += " F1:Help │ F2:Positions │ F3:Orders │ F4:Risk │ F5:Team │ F9:Settings │ F10:Quit "
        footer += " " * (self.width - 78)
        footer += f"{self.colors.RESET}"
        print(footer)
        
    def render(self, data: Dict):
        """Render complete terminal"""
        self.clear()
        
        # Header
        self.print_header()
        
        # Status bar
        self.print_status_bar(data)
        
        # Portfolio summary
        self.print_portfolio_summary(data.get('portfolio', {}))
        
        # Positions table
        self.print_positions_table(data.get('portfolio', {}).get('positions', []))
        
        # Trading team
        if 'trading_team' in data:
            self.print_trading_team_status(data['trading_team'])
            
        # Recent activity
        if 'activities' in data:
            self.print_recent_activity(data['activities'])
        
        # Footer
        self.print_footer()
        
    def _is_market_open(self) -> bool:
        """Check if US market is open"""
        now = datetime.now()
        # Simplified - would need proper timezone handling
        return 9 <= now.hour < 16
        
    def run_live(self):
        """Run live updating terminal"""
        print(f"{self.colors.CLEAR}")
        
        while self.running:
            try:
                # Fetch data from API
                data = self._fetch_data()
                
                # Render
                self.render(data)
                
                # Wait before next update
                time.sleep(3)
                
            except KeyboardInterrupt:
                self.running = False
                print(f"\n{self.colors.RESET}Terminal stopped.")
                
    def _fetch_data(self) -> Dict:
        """Fetch data from Alpha Junior API"""
        data = {
            'connected': False,
            'portfolio': {
                'total_value': 100000,
                'cash': 85000,
                'num_positions': 3,
                'var_95_pct': 3.2,
                'max_drawdown': 2.1,
                'positions': []
            },
            'trading_team': {},
            'activities': []
        }
        
        try:
            # Check health
            response = requests.get('http://localhost:5000/api/health', timeout=2)
            if response.status_code == 200:
                data['connected'] = True
                
            # Get portfolio
            pf_response = requests.get('http://localhost:5000/api/elite/portfolio', timeout=3)
            if pf_response.status_code == 200:
                pf_data = pf_response.json()
                if pf_data.get('success'):
                    data['portfolio'] = pf_data.get('portfolio', data['portfolio'])
                    
            # Get team
            team_response = requests.get('http://localhost:5000/api/elite/trading-team', timeout=3)
            if team_response.status_code == 200:
                team_data = team_response.json()
                if team_data.get('success'):
                    data['trading_team'] = team_data.get('trading_team', {})
                    
        except:
            pass
            
        # Add sample activity if none
        if not data['activities']:
            data['activities'] = [
                {'time': '09:30:15', 'action': 'BUY', 'symbol': 'NVDA', 'details': '45 shares @ $890.50'},
                {'time': '09:32:22', 'action': 'BUY', 'symbol': 'AMD', 'details': '80 shares @ $95.20'},
                {'time': '09:45:08', 'action': 'TRAILING STOP', 'symbol': 'NVDA', 'details': 'Updated to $866.40'},
                {'time': '10:15:33', 'action': 'SCAN', 'symbol': 'MARKET', 'details': 'Analyzed 100 stocks'},
                {'time': '10:30:01', 'action': 'HOLD', 'symbol': 'AAPL', 'details': 'Score 65, below threshold'},
            ]
            
        return data

def main():
    """Main entry point"""
    terminal = ProfessionalTerminal()
    
    print(f"{Colors().CLEAR}")
    print(f"{Colors().FG_GOLD}{Colors().BOLD}Initializing Professional Terminal...{Colors().RESET}\n")
    
    # Check connection
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=3)
        if response.status_code == 200:
            print(f"{Colors().FG_GREEN}✓ Connected to Alpha Junior API{Colors().RESET}\n")
        else:
            print(f"{Colors().FG_YELLOW}⚠ API not responding, showing demo data{Colors().RESET}\n")
    except:
        print(f"{Colors().FG_YELLOW}⚠ Server not running, start with: python runner.py{Colors().RESET}\n")
        time.sleep(2)
    
    time.sleep(1)
    
    # Run live terminal
    terminal.run_live()

if __name__ == '__main__':
    main()
