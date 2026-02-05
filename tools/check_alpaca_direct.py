import os
import requests
import json
from dotenv import load_dotenv

def check_alpaca():
    load_dotenv(override=True)

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")

    print(f"Checking Alpaca Connection to: {base_url}")
    print(f"API Key present: {bool(api_key)}")
    print(f"Secret Key present: {bool(secret_key)}")

    if not api_key or not secret_key:
        print("ERROR: Missing API Keys")
        return

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json"
    }

    # 1. Check Account
    try:
        r = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
        if r.status_code == 200:
            account = r.json()
            print("\n[ACCOUNT STATUS]")
            print(f"Status: {account.get('status')}")
            print(f"Equity: ${account.get('equity')}")
            print(f"Buying Power: ${account.get('buying_power')}")
            print(f"Cash: ${account.get('cash')}")
        else:
            print(f"ERROR: Account fetch failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"ERROR: Exception fetching account: {e}")

    # 2. Check Positions
    try:
        r = requests.get(f"{base_url}/v2/positions", headers=headers, timeout=10)
        if r.status_code == 200:
            positions = r.json()
            print(f"\n[POSITIONS] Count: {len(positions)}")
            for pos in positions:
                print(f"  - {pos['symbol']}: {pos['qty']} @ ${pos['current_price']} (P/L: {pos['unrealized_pl']})")
        else:
            print(f"ERROR: Positions fetch failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"ERROR: Exception fetching positions: {e}")

    # 3. Check Recent Orders
    try:
        params = {"status": "all", "limit": 10, "direction": "desc"}
        r = requests.get(f"{base_url}/v2/orders", headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            orders = r.json()
            print(f"\n[RECENT ORDERS] Last {len(orders)}")
            for o in orders:
                print(f"  - {o['submitted_at']} | {o['side'].upper()} {o['qty']} {o['symbol']} | Status: {o['status']}")
        else:
            print(f"ERROR: Orders fetch failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"ERROR: Exception fetching orders: {e}")

if __name__ == "__main__":
    check_alpaca()
