#!/usr/bin/env python3
"""
Setup Alpaca Trading Credentials
Stores API keys securely for trading
"""

import os
import sys
from pathlib import Path

def setup_alpaca():
    """Configure Alpaca API credentials."""
    
    print("\n" + "="*80)
    print("ALPACA TRADING SETUP")
    print("="*80)
    
    print("\n📍 Get your API credentials from: https://alpaca.markets/")
    print("   1. Sign up for Alpaca account (free)")
    print("   2. Go to Dashboard → API Keys")
    print("   3. Copy API Key and Secret Key")
    
    # Check if running in paper or live
    print("\n🔧 Which trading mode?")
    print("   1. Paper Trading (free simulation, recommended for testing)")
    print("   2. Live Trading (real money required)")
    
    mode = input("\nEnter choice (1 or 2): ").strip()
    
    if mode == "1":
        base_url = "https://paper-api.alpaca.markets"
        mode_name = "PAPER TRADING (Simulated)"
        print(f"\n✓ Using {mode_name}")
    elif mode == "2":
        base_url = "https://api.alpaca.markets"
        mode_name = "LIVE TRADING (Real Money)"
        print(f"\n⚠️  {mode_name} - Be careful with capital!")
    else:
        print("\n❌ Invalid choice")
        return False
    
    # Get API credentials
    print("\n📝 Enter your API credentials:")
    api_key = input("   API Key (APCA_API_KEY_ID): ").strip()
    secret_key = input("   Secret Key (APCA_API_SECRET_KEY): ").strip()
    
    if not api_key or not secret_key:
        print("\n❌ Missing credentials")
        return False
    
    # Validate credentials
    print("\n🔍 Validating credentials...")
    try:
        import alpaca_trade_api
        client = alpaca_trade_api.REST(api_key, secret_key, base_url)
        account = client.get_account()
        print(f"✓ Credentials valid!")
        print(f"  Account: {account.account_number}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")
        print(f"  Status: {account.status}")
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return False
    
    # Save to environment file
    env_file = Path(".env.alpaca")
    with open(env_file, "w") as f:
        f.write(f"APCA_API_KEY_ID={api_key}\n")
        f.write(f"APCA_API_SECRET_KEY={secret_key}\n")
        f.write(f"APCA_API_BASE_URL={base_url}\n")
    
    print(f"\n✓ Credentials saved to {env_file}")
    print("\nTo use these credentials when trading:")
    print(f"   export APCA_API_KEY_ID={api_key[:20]}...")
    print(f"   export APCA_API_SECRET_KEY=****")
    print(f"   export APCA_API_BASE_URL={base_url}")
    
    # Also save as batch file for Windows
    batch_file = Path("setup_alpaca_env.bat")
    with open(batch_file, "w") as f:
        f.write(f"@echo off\n")
        f.write(f"set APCA_API_KEY_ID={api_key}\n")
        f.write(f"set APCA_API_SECRET_KEY={secret_key}\n")
        f.write(f"set APCA_API_BASE_URL={base_url}\n")
        f.write(f"python complete_trading_system.py --mode live --broker alpaca\n")
    
    print(f"\n📦 Windows users: Run setup_alpaca_env.bat to start trading")
    
    print("\n" + "="*80)
    print("✓ SETUP COMPLETE")
    print("="*80)
    
    print("\n🚀 To start trading with Alpaca:")
    print("\n   Unix/Linux/Mac:")
    print(f"      export APCA_API_KEY_ID={api_key[:20]}...")
    print(f"      export APCA_API_SECRET_KEY=****")
    print(f"      python complete_trading_system.py --mode live --broker alpaca")
    
    print("\n   Windows (PowerShell):")
    print(f"      $env:APCA_API_KEY_ID = '{api_key[:20]}...'")
    print(f"      $env:APCA_API_SECRET_KEY = '****'")
    print(f"      python complete_trading_system.py --mode live --broker alpaca")
    
    print("\n   Windows (Command Prompt):")
    print(f"      setup_alpaca_env.bat")
    
    return True

if __name__ == "__main__":
    if setup_alpaca():
        sys.exit(0)
    else:
        sys.exit(1)
