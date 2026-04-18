"""
AUTO TRADING BOT - MiniQuantFund v4.0.0
Automatically places trades every minute - NO USER INPUT NEEDED
"""

import os
import sys
import time
import random
import logging
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/auto_trading.log')
    ]
)
logger = logging.getLogger('AutoTrader')

def main():
    """Auto trading - places trades immediately"""
    
    print("\n" + "="*70)
    print("🤖 AUTO TRADING BOT - STARTING IMMEDIATELY")
    print("="*70)
    print("⚠️  This will place REAL ORDERS in your Alpaca paper account!")
    print("Starting in 3 seconds...")
    print("="*70)
    
    time.sleep(3)
    
    # Load API keys
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ API keys not found!")
        return
    
    # Connect to Alpaca
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(api_key, secret_key,
                           'https://paper-api.alpaca.markets',
                           api_version='v2')
        
        account = api.get_account()
        print(f"✅ Connected! Equity: ${float(account.equity):,.2f}")
        print(f"✅ Buying Power: ${float(account.buying_power):,.2f}")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Trading parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    max_orders = 5  # Max orders to place
    orders_placed = []
    
    print("\n" + "🚀"*35)
    print("STARTING AUTOMATED TRADING")
    print("🚀"*35)
    
    try:
        for i in range(max_orders):
            try:
                # Pick symbol
                symbol = random.choice(symbols)
                
                # Get price
                try:
                    barset = api.get_barset([symbol], 'minute', limit=1)
                    if symbol in barset and len(barset[symbol]) > 0:
                        price = barset[symbol][0].close
                    else:
                        price = 100  # Default
                except:
                    price = 100
                
                # Calculate quantity (max $5K per trade)
                qty = max(1, min(10, int(5000 / price)))
                
                # Place BUY order
                print(f"\n📤 PLACING ORDER {i+1}/{max_orders}:")
                print(f"   Symbol: {symbol}")
                print(f"   Action: BUY")
                print(f"   Quantity: {qty} shares")
                print(f"   Estimated Price: ${price:.2f}")
                print(f"   Total Value: ${qty * price:.2f}")
                
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                print(f"✅ ORDER PLACED: {order.id}")
                orders_placed.append(order.id)
                
                # Wait for fill
                time.sleep(3)
                
                # Check status
                status = api.get_order(order.id)
                print(f"📊 Status: {status.status}")
                
                if status.status == 'filled':
                    print(f"✅ FILLED at ${status.filled_avg_price}")
                elif status.status == 'partially_filled':
                    print(f"⏳ Partial fill: {status.filled_qty}/{status.qty}")
                
                # Show positions
                positions = api.list_positions()
                if positions:
                    print(f"\n📊 Current Positions ({len(positions)}):")
                    for pos in positions:
                        pl = float(pos.unrealized_pl)
                        pl_emoji = "🟢" if pl >= 0 else "🔴"
                        print(f"   {pl_emoji} {pos.symbol}: {pos.qty} shares | "
                              f"P&L: ${pl:,.2f}")
                
                # Wait before next trade
                if i < max_orders - 1:
                    print(f"\n⏳ Waiting 10 seconds before next trade...")
                    time.sleep(10)
                
            except KeyboardInterrupt:
                print("\n\n⏹️  Trading stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error placing order: {e}")
                time.sleep(5)
        
        # Final summary
        print("\n" + "="*70)
        print("📈 TRADING SESSION COMPLETE")
        print("="*70)
        print(f"✅ Orders Placed: {len(orders_placed)}")
        
        # Show final positions
        positions = api.list_positions()
        print(f"📊 Final Positions: {len(positions)}")
        
        account = api.get_account()
        print(f"💰 Final Equity: ${float(account.equity):,.2f}")
        print(f"💳 Final Buying Power: ${float(account.buying_power):,.2f}")
        
        if positions:
            print(f"\n🏆 CHECK YOUR ALPACA DASHBOARD:")
            print(f"   https://paper-api.alpaca.markets/")
            print(f"   You should see {len(positions)} new positions!")
        
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Trading stopped")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
