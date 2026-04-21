#!/usr/bin/env python3
"""
Alpha Junior - ELITE HEDGE FUND ENGINE
Top 1% institutional-grade automated trading system
Combines AI traders, risk management, and portfolio optimization
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from institutional_traders_v2 import get_complete_team, TradeSignal, StrategyType
from institutional_portfolio import get_portfolio_manager, Position

class EliteHedgeFund:
    """
    Elite hedge fund trading engine
    Operates like Renaissance Technologies or Citadel
    """
    
    def __init__(self, alpaca_key: str, alpaca_secret: str):
        self.api_key = alpaca_key
        self.secret_key = alpaca_secret
        
        self.base_url = "https://paper-api.alpaca.markets"
        self.headers = {
            'APCA-API-KEY-ID': alpaca_key,
            'APCA-API-SECRET-KEY': alpaca_secret,
            'Content-Type': 'application/json'
        }
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize institutional components - 14 specialized traders
        self.trading_team = get_complete_team(self.logger)
        self.portfolio_manager = get_portfolio_manager(self.logger)
        
        # Trading state
        self.is_running = False
        self.scan_count = 0
        self.trades_executed = 0
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_realized_pnl = 0.0
        self.starting_capital = 100000.0
        
        # Market hours
        self.market_open = "09:30"
        self.market_close = "16:00"
        
        self.logger.info("=" * 80)
        self.logger.info("🎩 ELITE HEDGE FUND ENGINE INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info("Institutional-grade trading system activated")
        self.logger.info("14 specialized AI traders deployed")
        self.logger.info("Operating like Renaissance Technologies / Citadel")
        self.logger.info("=" * 80)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup institutional-grade logging"""
        logger = logging.getLogger('EliteHedgeFund')
        logger.setLevel(logging.INFO)
        
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            'logs/elite_hedge_fund.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console)
        
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
        """Get current positions from broker"""
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
    
    def get_stock_data_batch(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Get data for multiple stocks efficiently"""
        data = {}
        
        for symbol in symbols:
            try:
                response = requests.get(
                    f'https://data.alpaca.markets/v2/stocks/{symbol}/bars',
                    headers=self.headers,
                    params={'timeframe': '1Day', 'limit': 60, 'adjustment': 'split'},
                    timeout=5
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('bars'):
                        data[symbol] = result['bars']
                        
            except Exception as e:
                self.logger.debug(f"Error fetching {symbol}: {e}")
        
        return data
    
    def place_order(self, signal: TradeSignal) -> bool:
        """Execute institutional-grade order"""
        try:
            # Get account info for sizing
            account = self.get_account()
            portfolio_value = float(account.get('portfolio_value', 100000))
            
            # Get volatility for this stock
            data = self.get_stock_data_batch([signal.symbol])
            volatility = 25.0  # Default
            if signal.symbol in data and len(data[signal.symbol]) > 10:
                closes = [float(bar['c']) for bar in data[signal.symbol][-10:]]
                if len(closes) > 1:
                    volatility = np.std(closes) / np.mean(closes) * 100 * np.sqrt(252)
            
            # Calculate optimal position size using Kelly Criterion
            # Use conservative estimates until we have real track record
            win_rate = 0.60  # Target 60% win rate
            avg_win = signal.risk_reward_ratio * (signal.entry_price - signal.stop_loss)
            avg_loss = signal.entry_price - signal.stop_loss
            
            position_size = self.portfolio_manager.calculate_position_size(
                signal, portfolio_value, volatility
            )
            
            if position_size < 1:
                self.logger.info(f"⚠️ Position size too small for {signal.symbol}")
                return False
            
            # Check if we can add this position
            position_value = position_size * signal.entry_price
            can_add, reason = self.portfolio_manager.can_add_position(
                signal.symbol, position_value
            )
            
            if not can_add:
                self.logger.warning(f"🚫 Cannot add {signal.symbol}: {reason}")
                return False
            
            # Execute order
            order_data = {
                'symbol': signal.symbol,
                'qty': str(position_size),
                'side': signal.side,
                'type': 'market',
                'time_in_force': 'day'
            }
            
            response = requests.post(
                f'{self.base_url}/v2/orders',
                headers=self.headers,
                json=order_data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                order = response.json()
                
                self.logger.info("=" * 80)
                self.logger.info(f"🎯 EXECUTED: {signal.side.upper()} {position_size} {signal.symbol}")
                self.logger.info(f"   Strategy: {signal.strategy.value}")
                self.logger.info(f"   Score: {signal.score:.0f} | Confidence: {signal.confidence:.0f}%")
                self.logger.info(f"   Entry: ${signal.entry_price:.2f} | Target: ${signal.target_price:.2f} | Stop: ${signal.stop_loss:.2f}")
                self.logger.info(f"   R/R: {signal.risk_reward_ratio:.1f}:1")
                self.logger.info(f"   Reasoning: {signal.reasoning}")
                self.logger.info("=" * 80)
                
                # Add to portfolio tracking
                filled_price = float(order.get('filled_avg_price', signal.entry_price))
                self.portfolio_manager.positions[signal.symbol] = Position(
                    symbol=signal.symbol,
                    shares=position_size,
                    entry_price=filled_price,
                    current_price=filled_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.target_price,
                    strategy=signal.strategy.value,
                    entry_time=datetime.now(),
                    risk_amount=(filled_price - signal.stop_loss) * position_size,
                    portfolio_weight=(position_size * filled_price) / portfolio_value
                )
                
                self.portfolio_manager.cash -= position_size * filled_price
                self.trades_executed += 1
                
                return True
            else:
                self.logger.error(f"❌ Order failed: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
        
        return False
    
    def manage_positions(self):
        """Manage existing positions like institutional PMs"""
        positions = self.get_positions()
        
        if not positions:
            return
        
        self.logger.info("📊 Managing positions...")
        
        for symbol, pos in positions.items():
            try:
                current_price = float(pos.get('current_price', 0))
                entry_price = float(pos.get('avg_entry_price', current_price))
                qty = int(float(pos.get('qty', 0)))
                
                # Calculate P&L
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                self.logger.info(
                    f"   {symbol}: {qty} shares @ ${entry_price:.2f} → ${current_price:.2f} "
                    f"({unrealized_pnl_pct:+.2f}%)"
                )
                
                # Check stop loss / take profit
                if symbol in self.portfolio_manager.positions:
                    tracked_pos = self.portfolio_manager.positions[symbol]
                    
                    # Stop loss hit
                    if current_price <= tracked_pos.stop_loss:
                        self.logger.warning(f"🛑 STOP LOSS HIT: {symbol} at ${current_price:.2f}")
                        self.close_position(symbol, 'stop_loss')
                        continue
                    
                    # Take profit hit
                    if current_price >= tracked_pos.take_profit:
                        self.logger.info(f"🎯 TAKE PROFIT: {symbol} at ${current_price:.2f}")
                        self.close_position(symbol, 'take_profit')
                        continue
                    
                    # Trailing stop (advanced)
                    if unrealized_pnl_pct > 10:  # If up more than 10%
                        new_stop = current_price * 0.95  # Move stop to 5% below current
                        if new_stop > tracked_pos.stop_loss:
                            tracked_pos.stop_loss = new_stop
                            self.logger.info(f"📈 Trailing stop updated for {symbol}: ${new_stop:.2f}")
                
            except Exception as e:
                self.logger.warning(f"Error managing {symbol}: {e}")
    
    def close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            response = requests.delete(
                f'{self.base_url}/v2/positions/{symbol}',
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                # Calculate realized P&L
                if symbol in self.portfolio_manager.positions:
                    pos = self.portfolio_manager.positions[symbol]
                    pnl = pos.unrealized_pnl
                    self.total_realized_pnl += pnl
                    self.daily_pnl += pnl
                    
                    self.logger.info(
                        f"✅ CLOSED: {symbol} | Reason: {reason} | P&L: ${pnl:+.2f}"
                    )
                    
                    del self.portfolio_manager.positions[symbol]
                    
        except Exception as e:
            self.logger.error(f"Error closing {symbol}: {e}")
    
    def scan_opportunities(self):
        """Institutional-grade market scan"""
        self.scan_count += 1
        
        self.logger.info("=" * 80)
        self.logger.info(f"🔍 MARKET SCAN #{self.scan_count} - {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info("=" * 80)
        
        # Define stock universe (top liquid stocks)
        universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
            'AMD', 'CRM', 'ADBE', 'ORCL', 'INTC', 'UBER', 'ABNB', 'SNOW',
            'PLTR', 'ZM', 'SQ', 'ROKU', 'TWLO', 'DDOG', 'NET', 'CRWD',
            'JPM', 'BAC', 'V', 'MA', 'GS', 'MS', 'BLK', 'AXP',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT',
            'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'ENPH', 'SEDG', 'FSLR',
            'SHOP', 'COIN', 'MSTR', 'RIOT', 'MARA', 'HOOD', 'SOFI', 'AFRM',
            'AMC', 'GME', 'BB', 'NOK', 'EXPR', 'KOSS', 'NAKD'
        ]
        
        # Get data
        self.logger.info("📥 Fetching market data...")
        stock_data = self.get_stock_data_batch(universe)
        
        self.logger.info(f"📊 Analyzing {len(stock_data)} stocks with AI Team...")
        
        # Get consensus opportunities from trading team
        opportunities = self.trading_team.get_consensus_opportunities(stock_data)
        
        self.logger.info(f"🎯 Found {len(opportunities)} high-probability opportunities")
        
        # Execute best opportunities
        executed = 0
        for signal in opportunities[:5]:  # Top 5 opportunities
            if executed >= 3:  # Max 3 new positions per scan
                break
            
            # Check if we already own this
            if signal.symbol in self.portfolio_manager.positions:
                self.logger.info(f"⏭️ Already own {signal.symbol}, skipping")
                continue
            
            success = self.place_order(signal)
            if success:
                executed += 1
                time.sleep(1)  # Rate limiting
        
        # Manage existing positions
        self.manage_positions()
        
        # Portfolio summary
        self.print_portfolio_summary()
    
    def print_portfolio_summary(self):
        """Print institutional portfolio summary"""
        account = self.get_account()
        if account:
            portfolio_value = float(account.get('portfolio_value', 0))
            cash = float(account.get('cash', 0))
            
            pnl_pct = ((portfolio_value - self.starting_capital) / self.starting_capital) * 100
            
            self.logger.info("=" * 80)
            self.logger.info("📈 PORTFOLIO SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"   Portfolio Value: ${portfolio_value:,.2f} ({pnl_pct:+.2f}%)")
            self.logger.info(f"   Cash: ${cash:,.2f} ({(cash/portfolio_value)*100:.1f}%)")
            self.logger.info(f"   Positions: {len(self.portfolio_manager.positions)}")
            self.logger.info(f"   Daily P&L: ${self.daily_pnl:+.2f}")
            self.logger.info(f"   Total Realized P&L: ${self.total_realized_pnl:+.2f}")
            self.logger.info(f"   Scans Completed: {self.scan_count}")
            self.logger.info(f"   Trades Executed: {self.trades_executed}")
            self.logger.info("=" * 80)
    
    def run(self):
        """Main hedge fund trading loop"""
        self.logger.info("🚀 ELITE HEDGE FUND ENGINE STARTING...")
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Check market hours (simplified)
                current_time = datetime.now().strftime('%H:%M')
                
                if self.market_open <= current_time <= self.market_close:
                    # Market open - scan and trade
                    self.scan_opportunities()
                    
                    # Wait between scans
                    self.logger.info("⏳ Waiting 5 minutes for next scan...")
                    for _ in range(300):  # 300 seconds
                        if not self.is_running:
                            break
                        time.sleep(1)
                else:
                    # Market closed
                    self.logger.info("🌙 Market closed. Waiting for open...")
                    time.sleep(60)
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)
        
        self.logger.info("🛑 Elite Hedge Fund Engine stopped")
    
    def stop(self):
        """Stop the engine"""
        self.is_running = False
        self.logger.info("Stopping Elite Hedge Fund Engine...")

# Singleton
elite_engine = None

def get_elite_engine(alpaca_key: str = None, alpaca_secret: str = None) -> EliteHedgeFund:
    """Get elite hedge fund engine instance"""
    global elite_engine
    if elite_engine is None:
        key = alpaca_key or os.getenv('ALPACA_API_KEY', '')
        secret = alpaca_secret or os.getenv('ALPACA_SECRET_KEY', '')
        elite_engine = EliteHedgeFund(key, secret)
    return elite_engine

if __name__ == '__main__':
    # Standalone mode
    key = os.getenv('ALPACA_API_KEY', '')
    secret = os.getenv('ALPACA_SECRET_KEY', '')
    
    if not key or not secret:
        print("❌ Error: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        sys.exit(1)
    
    engine = EliteHedgeFund(key, secret)
    
    try:
        engine.run()
    except KeyboardInterrupt:
        engine.stop()
        print("\n✅ Elite Hedge Fund Engine stopped gracefully")
