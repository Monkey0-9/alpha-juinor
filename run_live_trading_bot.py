"""
LIVE TRADING BOT - MiniQuantFund v4.0.0
Actually places trades automatically like a quant fund
"""

import os
import sys
import time
import random
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/live_trading_bot.log')
    ]
)
logger = logging.getLogger('LiveTradingBot')

class LiveTradingBot:
    """Live trading bot that actually places trades"""
    
    def __init__(self):
        self.running = False
        self.api = None
        self.positions = {}
        self.daily_stats = {
            'orders_placed': 0,
            'orders_filled': 0,
            'total_volume': 0,
            'realized_pnl': 0
        }
        
        # Trading configuration
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        self.max_position_value = 10000  # $10K max per position
        self.max_orders_per_day = 10
        self.min_order_interval = 60  # seconds between orders
        
    def connect(self):
        """Connect to Alpaca"""
        load_dotenv()
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            logger.error("API keys not found")
            return False
        
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(api_key, secret_key,
                                     'https://paper-api.alpaca.markets',
                                     api_version='v2')
            
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca - Account: {account.status}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def get_market_data(self, symbol):
        """Get current market data for symbol"""
        try:
            # Get latest bar
            barset = self.api.get_barset([symbol], 'minute', limit=1)
            if symbol in barset and len(barset[symbol]) > 0:
                bar = barset[symbol][0]
                return {
                    'price': bar.close,
                    'volume': bar.volume,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low
                }
        except Exception as e:
            logger.warning(f"Could not get data for {symbol}: {e}")
        
        return None
    
    def generate_trade_signal(self, symbol, data):
        """Generate simple mean reversion signal"""
        if not data:
            return None
        
        price = data['price']
        
        # Simple strategy: Random mean reversion with position sizing
        # In production, this would use ML models, technical indicators, etc.
        
        # Check current position
        try:
            position = self.api.get_position(symbol)
            current_qty = int(position.qty)
        except:
            current_qty = 0
        
        # Simple logic: buy if no position, sell if has position
        # (This is just for demonstration - real strategies are more sophisticated)
        
        if current_qty == 0:
            # Buy signal
            qty = min(10, int(self.max_position_value / price))  # Max $10K position
            if qty > 0:
                return {'action': 'buy', 'qty': qty, 'price': price}
        else:
            # Sell signal (take profit or stop loss simulation)
            avg_entry = float(position.avg_entry_price)
            unrealized_pl = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc)
            
            # Sell if profit > 1% or loss > 2%
            if unrealized_plpc > 0.01 or unrealized_plpc < -0.02:
                return {'action': 'sell', 'qty': current_qty, 'price': price}
        
        return None
    
    def place_order(self, symbol, action, qty, price):
        """Place an order"""
        try:
            side = 'buy' if action == 'buy' else 'sell'
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day',
                client_order_id=f'bot_{datetime.now().strftime("%H%M%S")}_{symbol}'
            )
            
            logger.info(f"📤 ORDER PLACED: {side.upper()} {qty} {symbol} @ ~${price:.2f}")
            logger.info(f"   Order ID: {order.id}")
            
            self.daily_stats['orders_placed'] += 1
            self.daily_stats['total_volume'] += qty * price
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Order failed: {e}")
            return None
    
    def check_order_status(self, order_id):
        """Check and log order status"""
        try:
            order = self.api.get_order(order_id)
            
            if order.status == 'filled':
                logger.info(f"✅ ORDER FILLED: {order.symbol} {order.qty} shares")
                logger.info(f"   Filled Price: ${order.filled_avg_price}")
                self.daily_stats['orders_filled'] += 1
                return True
            elif order.status == 'partially_filled':
                logger.info(f"⏳ PARTIAL FILL: {order.filled_qty}/{order.qty}")
                return False
            else:
                logger.info(f"⏳ ORDER STATUS: {order.status}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking order: {e}")
            return False
    
    def update_positions(self):
        """Update and log current positions"""
        try:
            positions = self.api.list_positions()
            
            if positions:
                logger.info("\n📊 CURRENT POSITIONS:")
                total_value = 0
                total_pl = 0
                
                for pos in positions:
                    symbol = pos.symbol
                    qty = int(pos.qty)
                    avg_price = float(pos.avg_entry_price)
                    current_price = float(pos.current_price)
                    market_value = float(pos.market_value)
                    unrealized_pl = float(pos.unrealized_pl)
                    unrealized_plpc = float(pos.unrealized_plpc) * 100
                    
                    total_value += market_value
                    total_pl += unrealized_pl
                    
                    emoji = "🟢" if unrealized_pl >= 0 else "🔴"
                    logger.info(f"   {emoji} {symbol}: {qty} shares | "
                              f"Value: ${market_value:,.2f} | "
                              f"P&L: ${unrealized_pl:,.2f} ({unrealized_plpc:+.2f}%)")
                
                logger.info(f"   💰 Total Position Value: ${total_value:,.2f}")
                logger.info(f"   📈 Total Unrealized P&L: ${total_pl:,.2f}")
            else:
                logger.info("📊 No active positions")
            
            # Get account summary
            account = self.api.get_account()
            logger.info(f"\n💼 ACCOUNT SUMMARY:")
            logger.info(f"   Equity: ${float(account.equity):,.2f}")
            logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")
            logger.info(f"   Today's P&L: ${float(account.equity) - float(account.last_equity):,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def print_daily_stats(self):
        """Print daily trading statistics"""
        logger.info("\n" + "="*60)
        logger.info("📈 DAILY TRADING STATISTICS")
        logger.info("="*60)
        logger.info(f"   Orders Placed: {self.daily_stats['orders_placed']}")
        logger.info(f"   Orders Filled: {self.daily_stats['orders_filled']}")
        logger.info(f"   Total Volume: ${self.daily_stats['total_volume']:,.2f}")
        logger.info("="*60)
    
    def run(self):
        """Main trading loop"""
        logger.info("\n" + "🚀"*30)
        logger.info("LIVE TRADING BOT STARTED")
        logger.info("MiniQuantFund v4.0.0 - Automated Trading")
        logger.info("🚀"*30)
        
        if not self.connect():
            logger.error("Failed to connect. Exiting.")
            return False
        
        self.running = True
        order_count = 0
        last_trade_time = datetime.now() - timedelta(seconds=self.min_order_interval)
        
        try:
            while self.running:
                try:
                    current_time = datetime.now()
                    
                    # Only trade during market hours (simplified)
                    # Real system would check market hours properly
                    
                    # Check if enough time passed since last trade
                    if (current_time - last_trade_time).seconds >= self.min_order_interval:
                        
                        # Check daily order limit
                        if order_count < self.max_orders_per_day:
                            
                            # Pick a random symbol to trade
                            symbol = random.choice(self.symbols)
                            
                            # Get market data
                            data = self.get_market_data(symbol)
                            
                            if data:
                                # Generate trade signal
                                signal = self.generate_trade_signal(symbol, data)
                                
                                if signal:
                                    # Place the order
                                    order = self.place_order(
                                        symbol,
                                        signal['action'],
                                        signal['qty'],
                                        signal['price']
                                    )
                                    
                                    if order:
                                        order_count += 1
                                        last_trade_time = current_time
                                        
                                        # Wait and check status
                                        time.sleep(3)
                                        self.check_order_status(order.id)
                                        
                                        # Update positions display
                                        self.update_positions()
                    
                    # Print stats every 5 minutes
                    if current_time.minute % 5 == 0 and current_time.second < 10:
                        self.print_daily_stats()
                    
                    # Sleep to avoid hitting rate limits
                    time.sleep(10)
                    
                except KeyboardInterrupt:
                    logger.info("\n⏹️  Trading bot stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(30)  # Wait longer on error
        
        finally:
            self.print_daily_stats()
            logger.info("\n🏁 Trading bot shutdown complete")
        
        return True

def main():
    """Main function"""
    print("\n" + "="*70)
    print("🤖 MiniQuantFund v4.0.0 - Live Trading Bot")
    print("="*70)
    print("This bot will ACTUALLY PLACE TRADES in your Alpaca paper account!")
    print("It will buy and sell stocks automatically based on simple signals.")
    print("="*70)
    
    confirm = input("\n⚠️  Do you want to start automated trading? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("\n❌ Trading bot cancelled.")
        return
    
    bot = LiveTradingBot()
    
    try:
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
