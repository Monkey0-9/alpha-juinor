
import os
import json
import logging
import sys
from dotenv import load_dotenv

# Ensure root dir is in path
sys.path.append(os.getcwd())

# adapt to your alpaca client import
from brokers.alpaca_broker import AlpacaExecutionHandler

def main():
    load_dotenv(override=True)

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")

    print(f"--- Alpaca Account Diagnosis ---")
    print(f"URL: {base_url}")

    # Construction matching my modified AlpacaExecutionHandler
    # Note: my version uses requests.Session directly
    handler = AlpacaExecutionHandler(api_key, secret_key, base_url=base_url)

    try:
        acct = handler.get_account()
        print("=== Alpaca Account Info ===")
        print("status:", acct.get("status"))
        print("cash:", acct.get("cash"))
        print("buying_power:", acct.get("buying_power"))
        print("portfolio_value:", acct.get("portfolio_value"))
        print("trade_suspended_by_user:", acct.get("trade_suspended_by_user"))
        print("pattern_day_trader:", acct.get("pattern_day_trader"))
        print("shorting_enabled:", acct.get("shorting_enabled"))
        print("margin_balance:", acct.get("margin_balance"))
        print("multiplier:", acct.get("multiplier"))
    except Exception as e:
        print("Alpaca account check failed:", e)

if __name__ == "__main__":
    main()
