import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALPACA_API_KEY", "").strip()
api_secret = os.getenv("ALPACA_API_SECRET", "").strip()

print(f"Testing Key: [{api_key}] (len={len(api_key)})")
print(f"Testing Secret: [{api_secret}] (len={len(api_secret)})")

endpoints = [
    ("Paper API", "https://paper-api.alpaca.markets/v2/account"),
    ("Live API", "https://api.alpaca.markets/v2/account"),
    ("Data API", "https://data.alpaca.markets/v2/stocks/SPY/bars?timeframe=1Day&limit=1")
]

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": api_secret
}

for name, url in endpoints:
    try:
        res = requests.get(url, headers=headers)
        print(f"{name}: {res.status_code} - {res.text[:100]}")
    except Exception as e:
        print(f"{name}: Error {e}")
