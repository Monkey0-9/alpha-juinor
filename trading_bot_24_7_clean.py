"""
24/7 CONTINUOUS TRADING BOT - MiniQuantFund v4.0.0
Windows-compatible version without Unicode emojis
"""

import os
import sys
import time
import random
import logging
import signal
import threading
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trading_bot_24_7.log')
    ]
)
logger = logging.getLogger('TradingBot24_7')

class ContinuousTradingBot:
    """24/7 continuous trading bot"""
    
    def __init__(self):
        self.running = False
        self.api = None
        self.shutdown_event = threading.Event()
        
        # Trading stats
        self.stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'total_volume': 0,
            'start_time': datetime.now(),
            'last_trade_time': None
        }
        
        # Configuration
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM']
        self.max_position_value = 15000
        self.trade_interval = 300  # 5 minutes
        self.daily_trade_limit = 50
        
        # Risk management
        self.max_positions = 10
        self.max_daily_loss = -5000
        
    def signal_handler(self, signum, frame):
        """Handle shutdown gracefully"""
        logger.info("Shutdown signal received. Stopping bot gracefully...")
        self.running = False
        self.shutdown_event.set()
    
    def connect(self):
        """Connect to Alpaca"""
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
        """Check if market is open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except:
            return True
    
    def get_account_summary(self):
        """Get current account summary"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            last_equity = float(account.last_equity)
            daily_pl = equity - last_equity
            
            return {
                'equity': equity,
                'buying_power': buying_power,
                'daily_pl': daily_pl,
                'position_count': len(positions)
            }
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None
    
    def should_trade(self):
        """Determine if we should place a trade now"""
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
            return False, f"Daily trade limit reached: {self.stats['orders_placed']}"
        
        if self.stats['last_trade_time']:
            time_since_last = (datetime.now() - self.stats['last_trade_time']).seconds
            if time_since_last < self.trade_interval:
                return False, f"Waiting {self.trade_interval - time_since_last}s more"
        
        return True, "OK"
    
    def select_symbol_and_action(self):
        """Smart symbol and action selection"""
        try:
            positions = self.api.list_positions()
            position_symbols = {p.symbol: p for p in positions}
            
            # 60% chance to buy, 40% chance to sell if we have positions
            if positions and random.random() < 0.4:
                symbol = random.choice(list(position_symbols.keys()))
                pos = position_symbols[symbol]
                
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc)
                
                # Sell if profit > 2% or loss > 3%
                if unrealized_plpc > 0.02 or unrealized_plpc < -0.03:
                    qty = int(pos.qty)
                    return symbol, 'sell', qty
            
            # BUY
            available_symbols = [s for s in self.symbols if s not in position_symbols]
            if not available_symbols:
                available_symbols = self.symbols
            
            symbol = random.choice(available_symbols)
            
            # Get price for sizing
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
        """Place a trade"""
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
        """Log current portfolio status"""
        try:
            summary = self.get_account_summary()
            if not summary:
                return
            
            positions = self.api.list_positions()
            
            logger.info("\n" + "="*60)
            logger.info("PORTFOLIO STATUS UPDATE")
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
                    logger.info(f"   {status} {pos.symbol}: {pos.qty} shares | "
                              f"P&L: ${pl:,.2f} ({pl_pct:+.2f}%)")
                logger.info(f"   Total Unrealized P&L: ${total_pl:,.2f}")
            
            runtime = datetime.now() - self.stats['start_time']
            hours = runtime.seconds // 3600
            minutes = (runtime.seconds % 3600) // 60
            
            logger.info(f"\nRuntime: {hours}h {minutes}m")
            logger.info(f"Orders Today: {self.stats['orders_placed']}/{self.daily_trade_limit}")
            logger.info(f"Filled: {self.stats['orders_filled']}")
            logger.info(f"Volume Traded: ${self.stats['total_volume']:,.2f}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error logging portfolio: {e}")
    
    def run(self):
        """Main 24/7 trading loop"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("\n" + "="*70)
        logger.info("24/7 CONTINUOUS TRADING BOT")
        logger.info("MiniQuantFund v4.0.0")
        logger.info("="*70)
        logger.info("This bot will trade CONTINUOUSLY until stopped")
        logger.info("Press Ctrl+C to stop gracefully")
        logger.info("="*70)
        
        if not self.connect():
            logger.error("Failed to connect. Exiting.")
            return False
        
        self.running = True
        cycle_count = 0
        
        logger.info("\nBot started successfully!")
        logger.info("Waiting for trading opportunities...")
        
        try:
            while self.running and not self.shutdown_event.is_set():
                try:
                    cycle_count += 1
                    
                    # Log status every 5 minutes (30 cycles * 10 seconds)
                    if cycle_count % 30 == 0:
                        self.log_portfolio_status()
                    
                    # Check if we should trade
                    should_trade, reason = self.should_trade()
                    
                    if should_trade:
                        symbol, action, qty = self.select_symbol_and_action()
                        
                        if symbol and action and qty:
                            success = self.place_trade(symbol, action, qty)
                            
                            if success:
                                logger.info(f"Trade cycle {cycle_count} complete")
                            else:
                                logger.warning(f"Trade cycle {cycle_count} failed")
                        else:
                            logger.info(f"No trade opportunity this cycle")
                    else:
                        if cycle_count % 10 == 0:  # Log reason every 10 cycles
                            logger.info(f"{reason}")
                    
                    # Sleep until next cycle (10 seconds)
                    time.sleep(10)
                    
                except KeyboardInterrupt:
                    logger.info("\nKeyboard interrupt received")
                    break
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    time.sleep(30)
        
        finally:
            logger.info("\n" + "="*60)
            logger.info("BOT SHUTDOWN")
            logger.info("="*60)
            self.log_portfolio_status()
            logger.info("Trading bot stopped gracefully")
            logger.info("="*60)
        
        return True

def main():
    """Main entry point"""
    bot = ContinuousTradingBot()
    
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
