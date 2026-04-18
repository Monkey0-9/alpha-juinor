#!/usr/bin/env python3
"""
Test Alpaca connection and credentials
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlpacaTest")

def test_alpaca():
    """Test Alpaca connection."""
    
    print("\n" + "="*80)
    print("ALPACA CONNECTION TEST")
    print("="*80)
    
    # Check if SDK installed
    try:
        import alpaca_trade_api
        logger.info("✓ alpaca-trade-api SDK installed")
    except ImportError:
        logger.error("✗ alpaca-trade-api SDK not installed")
        logger.error("  Install with: pip install alpaca-trade-api")
        return False
    
    # Check credentials
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not api_key or not secret_key:
        logger.warning("✗ Credentials not set in environment")
        logger.info("  Run: python setup_alpaca.py")
        
        # Check .env.alpaca file
        if os.path.exists(".env.alpaca"):
            logger.info("  Found .env.alpaca - loading credentials...")
            with open(".env.alpaca", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        os.environ[key] = value
                        if "SECRET" in key:
                            logger.info(f"  Set {key}=****")
                        else:
                            logger.info(f"  Set {key}={value}")
            
            api_key = os.getenv("APCA_API_KEY_ID")
            secret_key = os.getenv("APCA_API_SECRET_KEY")
        else:
            return False
    else:
        logger.info(f"✓ API Key found: {api_key[:10]}...")
        logger.info(f"✓ Secret Key found: {secret_key[:10]}...")
    
    logger.info(f"✓ Base URL: {base_url}")
    
    # Test connection
    logger.info("\n🔍 Testing connection...")
    try:
        client = alpaca_trade_api.REST(api_key, secret_key, base_url)
        account = client.get_account()
        
        logger.info("✓ Connection successful!")
        logger.info(f"  Account: {account.account_number}")
        logger.info(f"  Status: {account.status}")
        logger.info(f"  Equity: ${float(account.equity):,.2f}")
        logger.info(f"  Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"  Cash: ${float(account.cash):,.2f}")
        
        # Test market data
        logger.info("\n📊 Testing market data...")
        bars = client.get_bar_set(["AAPL", "MSFT", "GOOGL"], "minute", limit=1)
        if bars:
            logger.info("✓ Market data available")
            for symbol in ["AAPL", "MSFT", "GOOGL"]:
                if symbol in bars:
                    bar = bars[symbol].iloc[-1]
                    logger.info(f"  {symbol}: ${bar.c:.2f}")
        
        print("\n" + "="*80)
        print("✓ ALPACA READY FOR TRADING")
        print("="*80)
        print("\nTo start trading:")
        print("  python complete_trading_system.py --mode paper --broker alpaca")
        print("\nTo trade with real money (after validation):")
        print("  python complete_trading_system.py --mode live --broker alpaca --capital 1000")
        print()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Connection failed: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check API credentials are correct")
        logger.error("  2. Check internet connection")
        logger.error("  3. Check Alpaca status: https://status.alpaca.markets/")
        logger.error("  4. Re-run setup: python setup_alpaca.py")
        return False

if __name__ == "__main__":
    if test_alpaca():
        sys.exit(0)
    else:
        sys.exit(1)
