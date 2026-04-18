"""
24/7 TRADING BOT - Windows Compatible Version
MiniQuantFund v4.0.0
"""

import os
import sys
import time
import random
import logging
import signal
import threading
import io
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
os.makedirs('logs', exist_ok=True)

# Create logger
logger = logging.getLogger('TradingBot')
logger.setLevel(logging.INFO)

# File handler (works fine)
file_handler = logging.FileHandler('logs/trading_bot_24_7.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Console handler with utf-8
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Simple formatter
formatter = logging.Formatter('%(asctime)s | %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

class TradingBot:
    """Continuous trading bot"""
    
    def __init__(self):
        self.running = False
        self.api = None
        self.shutdown_event = threading.Event()
        self.stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'total_volume': 0,
            'start_time': datetime.now(),
            'last_trade_time': None
        }
        
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM']
        self.max_position_value = 15000
        self.trade_interval = 300
        self.daily_trade_limit = 50
        self.max_positions = 10
        self.max_daily_loss = -5000
        
    def signal_handler(self, signum, frame):
        logger.info("Shutdown signal received. Stopping bot...")
        self.running = False
        self.shutdown_event.set()
    
    def connect(self):
        load_dotenv()
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            logger.error("ERROR: API keys not found!")
            return False
        
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(api_key, secret_key,
                                     'https://paper-api.alpaca.markets',
                                     api_version='v2')
            
            account = self.api.get_account()
            logger.info("Connected to Alpaca Paper Trading")
            logger.info(f"Equity: ${float(account.equity):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def is_market_open(self):
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except:
            return True
    
    def get_account_summary(self):
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'daily_pl': float(account.equity) - float(account.last_equity),
                'position_count': len(positions)
            }
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None
    
    def should_trade(self):
        if not self.is_market_open():
            return False, "Market closed"
        
        summary = self.get_account_summary()
        if not summary:
            return False, "Can't get account data"
        
        if summary['daily_pl'] < self.max_daily_loss:
            return False, f"Daily loss limit hit: ${summary['daily_pl']:,.2f}"
        
        if summary['position_count'] >= self.max_positions:
            return False, f"Max positions reached: {summary['position_count']}"
        
        if self.stats['orders_placed'] >= self.daily_trade_limit:
            return False, f"Daily trade limit: {self.stats['orders_placed']}"
        
        if self.stats['last_trade_time']:
            time_since_last = (datetime.now() - self.stats['last_trade_time']).seconds
            if time_since_last < self.trade_interval:
                return False, f"Waiting {self.trade_interval - time_since_last}s"
        
        return True, "OK"
    
    def select_symbol_and_action(self):
        try:
            positions = self.api.list_positions()
            position_symbols = {p.symbol: p for p in positions}
            
            if positions and random.random() < 0.4:
                symbol = random.choice(list(position_symbols.keys()))
                pos = position_symbols[symbol]
                
                unrealized_plpc = float(pos.unrealized_plpc)
                
                if unrealized_plpc > 0.02 or unrealized_plpc < -0.03:
                    return symbol, 'sell', int(pos.qty)
            
            available_symbols = [s for s in self.symbols if s not in position_symbols]
            if not available_symbols:
                available_symbols = self.symbols
            
            symbol = random.choice(available_symbols)
            
            try:
                barset = self.api.get_barset([symbol], 'minute', limit=1)
                if symbol in barset and len(barset[symbol]) > 0:
                    price = barset[symbol][0].close
                    qty = max(1, min(15, int(self.max_position_value / price)))
                else:
                    qty = random.randint(5, 15)
            except:
                qty = random.randint(5, 15)
            
            return symbol, 'buy', qty
            
        except Exception as e:
            logger.error(f"Error selecting trade: {e}")
            return None, None, None
    
    def place_trade(self, symbol, action, qty):
        try:
            side = 'buy' if action == 'buy' else 'sell'
            
            logger.info(f"\nPLACING {side.upper()} ORDER:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Quantity: {qty} shares")
            logger.info(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            logger.info(f"ORDER PLACED: {order.id}")
            
            self.stats['orders_placed'] += 1
            self.stats['last_trade_time'] = datetime.now()
            
            time.sleep(3)
            
            status = self.api.get_order(order.id)
            if status.status == 'filled':
                fill_price = float(status.filled_avg_price)
                value = fill_price * int(status.filled_qty)
                self.stats['orders_filled'] += 1
                self.stats['total_volume'] += value
                logger.info(f"FILLED: {status.filled_qty} shares @ ${fill_price:.2f} = ${value:,.2f}")
                return True
            elif status.status == 'partially_filled':
                logger.info(f"PARTIAL: {status.filled_qty}/{status.qty}")
                return True
            else:
                logger.info(f"STATUS: {status.status}")
                return True
                
        except Exception as e:
            logger.error(f"Trade failed: {e}")
            return False
    
    def log_portfolio_status(self):
        try:
            summary = self.get_account_summary()
            if not summary:
                return
            
            positions = self.api.list_positions()
            
            logger.info("\n" + "="*60)
            logger.info("PORTFOLIO STATUS")
            logger.info("="*60)
            logger.info(f"Equity: ${summary['equity']:,.2f}")
            logger.info(f"Buying Power: ${summary['buying_power']:,.2f}")
            logger.info(f"Daily P&L: ${summary['daily_pl']:,.2f}")
            logger.info(f"Positions: {summary['position_count']}/{self.max_positions}")
            
            if positions:
                total_pl = 0
                for pos in positions:
                    pl = float(pos.unrealized_pl)
                    pl_pct = float(pos.unrealized_plpc) * 100
                    total_pl += pl
                    status = "PROFIT" if pl >= 0 else "LOSS"
                    logger.info(f"   {status} {pos.symbol}: {pos.qty} shares | P&L: ${pl:,.2f} ({pl_pct:+.2f}%)")
                logger.info(f"   Total Unrealized P&L: ${total_pl:,.2f}")
            
            runtime = datetime.now() - self.stats['start_time']
            hours = runtime.seconds // 3600
            minutes = (runtime.seconds % 3600) // 60
            
            logger.info(f"\nRuntime: {hours}h {minutes}m")
            logger.info(f"Orders Today: {self.stats['orders_placed']}/{self.daily_trade_limit}")
            logger.info(f"Filled: {self.stats['orders_filled']}")
            logger.info(f"Volume: ${self.stats['total_volume']:,.2f}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error logging portfolio: {e}")
    
    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("\n" + "="*70)
        print("24/7 TRADING BOT - MiniQuantFund v4.0.0")
        print("="*70)
        print("This bot trades CONTINUOUSLY until stopped")
        print("Press Ctrl+C to stop")
        print("="*70)
        
        if not self.connect():
            print("ERROR: Failed to connect. Exiting.")
            return False
        
        self.running = True
        cycle_count = 0
        
        print("\nBot started successfully!")
        print("Waiting for trading opportunities...")
        print("(Check logs/trading_bot_24_7.log for detailed output)")
        
        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    cycle_count += 1
                    
                    if cycle_count % 30 == 0:
                        self.log_portfolio_status()
                    
                    should_trade, reason = self.should_trade()
                    
                    if should_trade:
                        symbol, action, qty = self.select_symbol_and_action()
                        
                        if symbol and action and qty:
                            success = self.place_trade(symbol, action, qty)
                            
                            if success:
                                logger.info(f"Trade cycle {cycle_count} complete")
                        else:
                            logger.info(f"No trade opportunity")
                    else:
                        if cycle_count % 10 == 0:
                            logger.info(f"{reason}")
                    
                    time.sleep(10)
                    
                except KeyboardInterrupt:
                    print("\nStopping...")
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")
                    time.sleep(30)
        
        finally:
            print("\n" + "="*60)
            print("BOT SHUTDOWN")
            print("="*60)
            self.log_portfolio_status()
            print("Bot stopped gracefully")
            print("="*60)
        
        return True

def main():
    bot = TradingBot()
    
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
