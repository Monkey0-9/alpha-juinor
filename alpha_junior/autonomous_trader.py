#!/usr/bin/env python3
"""
Alpha Junior - AUTONOMOUS TRADER
Fully automated system that scans ALL stocks and trades dynamically
Uses AI Brain to pick the best opportunities 24/7
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List
import requests

from brain import AlphaBrain, get_brain

class AutonomousTrader:
    """
    Fully autonomous trading system
    - Scans 100+ stocks every 5 minutes
    - Uses AI Brain to score opportunities
    - Automatically buys high-score stocks
    - Automatically sells low-score positions
    - Manages portfolio risk
    - Targets 50-60% annual returns
    """
    
    def __init__(self, alpaca_key: str, alpaca_secret: str, 
                 paper_trading: bool = True):
        self.api_key = alpaca_key
        self.secret_key = alpaca_secret
        self.paper_trading = paper_trading
        
        self.base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        # Initialize AI Brain
        self.brain = get_brain(alpaca_key, alpaca_secret)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Trading parameters
        self.params = {
            'max_positions': 20,           # Max simultaneous positions
            'max_position_size': 10000,    # Max $ per position
            'min_position_size': 1000,     # Min $ per position
            'buy_threshold': 75,           # Brain score to trigger buy
            'sell_threshold': 40,          # Brain score to trigger sell
            'stop_loss_pct': 8,            # Stop loss percentage
            'take_profit_pct': 20,         # Take profit percentage
            'scan_interval': 300,          # Seconds between scans (5 min)
            'max_daily_trades': 50,       # Max trades per day
        }
        
        # State tracking
        self.is_running = False
        self.positions = {}  # Current positions
        self.daily_stats = {
            'trades_today': 0,
            'buys_today': 0,
            'sells_today': 0,
            'daily_pnl': 0,
            'last_reset': datetime.now().date()
        }
        self.trade_history = []
        
        # Performance tracking
        self.total_invested = 0
        self.total_realized_pnl = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('AutonomousTrader')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        # File handler
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        file_handler = logging.FileHandler('logs/autonomous_trader.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def get_account(self) -> Dict:
        """Get account information"""
        try:
            response = requests.get(
                f'{self.base_url}/v2/account',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.logger.error(f"Error getting account: {e}")
        return {}
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
        try:
            response = requests.get(
                f'{self.base_url}/v2/positions',
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                positions = response.json()
                return {p['symbol']: p for p in positions}
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
        return {}
    
    def place_order(self, symbol: str, qty: int, side: str, 
                    order_type: str = 'market') -> Dict:
        """Place an order"""
        try:
            data = {
                'symbol': symbol,
                'qty': str(qty),
                'side': side,
                'type': order_type,
                'time_in_force': 'day'
            }
            
            response = requests.post(
                f'{self.base_url}/v2/orders',
                headers=self.headers,
                json=data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                order = response.json()
                self.logger.info(f"✅ ORDER FILLED: {side.upper()} {qty} {symbol}")
                return order
            else:
                self.logger.error(f"❌ Order failed: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
        return None
    
    def close_position(self, symbol: str) -> bool:
        """Close a position"""
        try:
            response = requests.delete(
                f'{self.base_url}/v2/positions/{symbol}',
                headers=self.headers,
                timeout=10
            )
            if response.status_code in [200, 204]:
                self.logger.info(f"✅ CLOSED: {symbol}")
                return True
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
        return False
    
    def calculate_position_size(self, price: float, score: float, 
                                 available_cash: float) -> int:
        """
        Calculate optimal position size based on:
        - Available cash
        - Stock score (higher = bigger position)
        - Risk management
        """
        # Base size on score
        if score >= 90:
            allocation_pct = 0.15  # 15% of available cash
        elif score >= 85:
            allocation_pct = 0.12
        elif score >= 80:
            allocation_pct = 0.10
        elif score >= 75:
            allocation_pct = 0.08
        else:
            allocation_pct = 0.05
        
        # Calculate dollar amount
        max_dollar = min(
            available_cash * allocation_pct,
            self.params['max_position_size']
        )
        
        # Ensure minimum
        if max_dollar < self.params['min_position_size']:
            return 0
        
        # Calculate shares
        shares = int(max_dollar / price)
        
        # Cap at reasonable limits
        if shares < 1:
            return 0
        elif shares > 100 and price < 10:
            return 100  # Avoid too many penny stocks
        
        return shares
    
    def check_stop_loss_take_profit(self, position: Dict) -> str:
        """
        Check if position should be closed due to stop loss or take profit
        Returns: 'hold', 'stop_loss', or 'take_profit'
        """
        try:
            current_price = float(position.get('current_price', 0))
            entry_price = float(position.get('avg_entry_price', current_price))
            
            if entry_price == 0:
                return 'hold'
            
            # Calculate unrealized P&L %
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Check stop loss
            if pnl_pct <= -self.params['stop_loss_pct']:
                return 'stop_loss'
            
            # Check take profit
            if pnl_pct >= self.params['take_profit_pct']:
                return 'take_profit'
            
        except Exception as e:
            self.logger.warning(f"Error checking position: {e}")
        
        return 'hold'
    
    def scan_and_trade(self):
        """
        Main trading loop - scans market and executes trades
        """
        self.logger.info("=" * 70)
        self.logger.info("🧠 BRAIN: Starting market scan...")
        self.logger.info("=" * 70)
        
        # 1. Get account status
        account = self.get_account()
        if not account:
            self.logger.error("❌ Cannot get account info - skipping scan")
            return
        
        cash = float(account.get('cash', 0))
        portfolio_value = float(account.get('portfolio_value', 0))
        buying_power = float(account.get('buying_power', 0))
        
        self.logger.info(f"💰 Account: Cash=${cash:,.2f} | Portfolio=${portfolio_value:,.2f}")
        
        # 2. Get current positions
        positions = self.get_positions()
        self.positions = positions
        
        self.logger.info(f"📊 Current positions: {len(positions)}")
        
        # 3. Check existing positions for sell signals
        self.logger.info("🔍 Checking current positions for SELL signals...")
        
        for symbol, position in positions.items():
            # Check stop loss / take profit first
            action = self.check_stop_loss_take_profit(position)
            
            if action == 'stop_loss':
                self.logger.warning(f"🛑 STOP LOSS triggered for {symbol}")
                if self.close_position(symbol):
                    self.daily_stats['sells_today'] += 1
                    self.daily_stats['trades_today'] += 1
                continue
            
            if action == 'take_profit':
                self.logger.info(f"🎯 TAKE PROFIT triggered for {symbol}")
                if self.close_position(symbol):
                    self.daily_stats['sells_today'] += 1
                    self.daily_stats['trades_today'] += 1
                continue
            
            # Check brain score for position
            stock_data = self.brain.get_stock_data([symbol], days=10)
            if symbol in stock_data:
                indicators = self.brain.calculate_indicators(stock_data[symbol])
                if indicators:
                    score = self.brain.calculate_brain_score(indicators)
                    self.logger.info(f"  {symbol}: Score={score:.1f} | Price=${indicators['price']:.2f}")
                    
                    # Sell if score drops below threshold
                    if score <= self.params['sell_threshold']:
                        self.logger.info(f"📉 SELL SIGNAL: {symbol} score dropped to {score:.1f}")
                        if self.close_position(symbol):
                            self.daily_stats['sells_today'] += 1
                            self.daily_stats['trades_today'] += 1
        
        # 4. Scan for new opportunities
        available_slots = self.params['max_positions'] - len(positions)
        
        if available_slots > 0 and cash > self.params['min_position_size']:
            self.logger.info(f"🔍 Scanning for new BUY opportunities...")
            self.logger.info(f"   Available slots: {available_slots} | Cash: ${cash:,.2f}")
            
            # Use AI Brain to find top picks
            top_picks = self.brain.get_top_picks(n=20, min_score=self.params['buy_threshold'])
            
            # Filter out stocks we already own
            new_opportunities = [p for p in top_picks if p['symbol'] not in positions]
            
            self.logger.info(f"   Found {len(new_opportunities)} new opportunities")
            
            # Buy top opportunities
            for pick in new_opportunities[:available_slots]:
                if self.daily_stats['trades_today'] >= self.params['max_daily_trades']:
                    self.logger.warning("⚠️ Daily trade limit reached")
                    break
                
                symbol = pick['symbol']
                score = pick['score']
                price = pick['price']
                
                # Calculate position size
                shares = self.calculate_position_size(price, score, cash)
                
                if shares > 0:
                    # Check if we have enough cash
                    cost = shares * price
                    if cost <= cash:
                        self.logger.info(f"🚀 BUY SIGNAL: {symbol}")
                        self.logger.info(f"   Score: {score:.1f} | Shares: {shares} | Cost: ${cost:,.2f}")
                        self.logger.info(f"   Reason: {pick['recommendation']}")
                        
                        # Place buy order
                        order = self.place_order(symbol, shares, 'buy')
                        if order:
                            cash -= cost
                            self.daily_stats['buys_today'] += 1
                            self.daily_stats['trades_today'] += 1
                            self.total_invested += cost
                            
                            # Record trade
                            self.trade_history.append({
                                'symbol': symbol,
                                'side': 'buy',
                                'shares': shares,
                                'price': price,
                                'cost': cost,
                                'score': score,
                                'timestamp': datetime.now()
                            })
        else:
            if available_slots <= 0:
                self.logger.info("ℹ️ Max positions reached - no new buys")
            if cash <= self.params['min_position_size']:
                self.logger.info(f"ℹ️ Insufficient cash (${cash:,.2f}) - no new buys")
        
        # 5. Generate summary
        self.logger.info("=" * 70)
        self.logger.info("📈 TRADING SESSION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"   Total positions: {len(self.get_positions())}")
        self.logger.info(f"   Trades today: {self.daily_stats['trades_today']}")
        self.logger.info(f"   Buys today: {self.daily_stats['buys_today']}")
        self.logger.info(f"   Sells today: {self.daily_stats['sells_today']}")
        self.logger.info(f"   Cash available: ${cash:,.2f}")
        self.logger.info("=" * 70)
    
    def reset_daily_stats(self):
        """Reset daily statistics at market open"""
        today = datetime.now().date()
        if today != self.daily_stats['last_reset']:
            self.logger.info("🌅 New trading day - resetting statistics")
            self.daily_stats = {
                'trades_today': 0,
                'buys_today': 0,
                'sells_today': 0,
                'daily_pnl': 0,
                'last_reset': today
            }
    
    def run(self):
        """Main loop - runs continuously"""
        self.logger.info("=" * 70)
        self.logger.info("🤖 AUTONOMOUS TRADER STARTED")
        self.logger.info("=" * 70)
        self.logger.info(f"Parameters:")
        self.logger.info(f"  Max positions: {self.params['max_positions']}")
        self.logger.info(f"  Buy threshold: {self.params['buy_threshold']}")
        self.logger.info(f"  Sell threshold: {self.params['sell_threshold']}")
        self.logger.info(f"  Scan interval: {self.params['scan_interval']} seconds")
        self.logger.info(f"  Stop loss: {self.params['stop_loss_pct']}%")
        self.logger.info(f"  Take profit: {self.params['take_profit_pct']}%")
        self.logger.info("=" * 70)
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Reset daily stats if new day
                self.reset_daily_stats()
                
                # Run trading logic
                self.scan_and_trade()
                
                # Wait before next scan
                self.logger.info(f"⏳ Waiting {self.params['scan_interval']} seconds for next scan...")
                
                # Sleep with interrupt check
                for _ in range(self.params['scan_interval']):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
        
        self.logger.info("🛑 Autonomous trader stopped")
    
    def stop(self):
        """Stop the trader"""
        self.is_running = False
        self.logger.info("Stopping autonomous trader...")
    
    def get_status(self) -> Dict:
        """Get current status"""
        positions = self.get_positions()
        account = self.get_account()
        
        return {
            'is_running': self.is_running,
            'positions_count': len(positions),
            'cash': float(account.get('cash', 0)) if account else 0,
            'portfolio_value': float(account.get('portfolio_value', 0)) if account else 0,
            'daily_trades': self.daily_stats['trades_today'],
            'daily_buys': self.daily_stats['buys_today'],
            'daily_sells': self.daily_stats['sells_today'],
            'total_invested': self.total_invested,
            'trade_history_count': len(self.trade_history),
            'brain_stats': self.brain.get_brain_stats()
        }

# Singleton instance
_trader_instance = None

def get_trader(alpaca_key: str = None, alpaca_secret: str = None) -> AutonomousTrader:
    """Get or create trader instance"""
    global _trader_instance
    if _trader_instance is None:
        key = alpaca_key or os.getenv('ALPACA_API_KEY', '')
        secret = alpaca_secret or os.getenv('ALPACA_SECRET_KEY', '')
        _trader_instance = AutonomousTrader(key, secret)
    return _trader_instance

if __name__ == '__main__':
    # Standalone mode
    key = os.getenv('ALPACA_API_KEY', '')
    secret = os.getenv('ALPACA_SECRET_KEY', '')
    
    if not key or not secret:
        print("❌ Error: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        sys.exit(1)
    
    trader = AutonomousTrader(key, secret)
    
    try:
        trader.run()
    except KeyboardInterrupt:
        trader.stop()
        print("\n✅ Trader stopped gracefully")
