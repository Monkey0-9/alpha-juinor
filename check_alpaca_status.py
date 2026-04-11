#!/usr/bin/env python
"""Check Alpaca trading account status"""
import os
import sys
from pprint import pprint

try:
    from alpaca.broker_client import BrokerClient

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        print("ERROR: Alpaca API credentials not configured")
        print(f"  APCA_API_KEY_ID: {bool(api_key)}")
        print(f"  APCA_API_SECRET_KEY: {bool(secret_key)}")
        sys.exit(1)

    client = BrokerClient(
        api_key=api_key,
        secret_key=secret_key,
        base_url="https://broker-api.sandbox.alpaca.markets",
    )

    # Get account info
    account = client.get_account()
    print("=== ALPACA ACCOUNT STATUS ===")
    print(f"Account Status: {account.account_status}")
    print(f"Trading Status: {account.trading_status}")
    print(f"Account Equity: ${account.equity:,.2f}")
    print(f"Buying Power: ${account.buying_power:,.2f}")
    print(f"Positions Count: {len(account.positions)}")

    # List positions
    if account.positions:
        print(f"\n=== OPEN POSITIONS ({len(account.positions)}) ===")
        for pos in account.positions:
            print(f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_fill_price:.2f}")
    else:
        print("\nNo open positions")

except Exception as e:
    import traceback

    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)
