import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_API_SECRET")
paper = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"

base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
data_url = "https://data.alpaca.markets"

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": api_secret
}

print(f"Testing keys: {api_key[:5]}... / {api_secret[:5]}...")
print(f"Paper: {paper}")

try:
    # Test account
    acc_res = requests.get(f"{base_url}/v2/account", headers=headers)
    print(f"Account Response: {acc_res.status_code}")
    print(acc_res.json() if acc_res.status_code == 200 else acc_res.text)

    # Test bars
    bars_res = requests.get(f"{data_url}/v2/stocks/SPY/bars?timeframe=1Day&limit=1", headers=headers)
    print(f"Bars Response: {bars_res.status_code}")
    print(bars_res.json() if bars_res.status_code == 200 else bars_res.text)
except Exception as e:
    print(f"Error: {e}")
