"""
Elite Quant Fund System - Runner Script
Execute world-class quantitative trading system
"""

import asyncio
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/elite_quant_fund_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)

# Ensure logs directory exists
import os
os.makedirs('logs', exist_ok=True)

from elite_quant_fund import create_elite_quant_fund


async def main():
    """Main entry point"""
    
    print("\n" + "="*80)
    print("ELITE QUANT FUND SYSTEM v1.0.0")
    print("World-Class Quantitative Trading")
    print("="*80)
    print("Initializing system...")
    print("="*80 + "\n")
    
    # Create and configure system
    system = create_elite_quant_fund(
        symbols=[
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
            'META', 'TSLA', 'NFLX', 'AMD', 'CRM'
        ],
        initial_capital=10_000_000,  # $10M
        target_volatility=0.10,      # 10% target vol
        kelly_fraction=0.3           # Conservative Kelly
    )
    
    try:
        # Start system
        await system.start()
        
        print("\n" + "="*80)
        print("SYSTEM RUNNING - Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        # Run indefinitely
        while system._running:
            await asyncio.sleep(1)
            
            # Print status every 60 seconds
            if int(datetime.now().timestamp()) % 60 == 0:
                status = system.get_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Portfolio: ${status['portfolio_value']:>15,.2f} | "
                      f"P&L: ${status['portfolio_pnl']:>12,.2f} | "
                      f"DD: {status['current_drawdown']:>6.2%} | "
                      f"Signals: {status['signals_generated']:>5}")
    
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    
    finally:
        # Stop system
        await system.stop()
        
        print("\n" + "="*80)
        print("SYSTEM STOPPED")
        print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
