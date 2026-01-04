import os
import requests
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("ALPACA_API_KEY")
secret = os.getenv("ALPACA_SECRET_KEY")
headers = {
    "APCA-API-KEY-ID": key,
    "APCA-API-SECRET-KEY": secret
}

# 1. Test Stock
ticker = "AAPL"
url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
params = {
    "start": "2023-01-01",
    "end": "2023-01-10",
    "timeframe": "1Day"
}
print(f"Testing Stock {ticker}...")
r = requests.get(url, headers=headers, params=params)
print(f"Status: {r.status_code}")
if r.status_code != 200:
    print(f"Error: {r.text}")

# 2. Test Crypto
ticker = "BTC/USD"
url = f"https://data.alpaca.markets/v1beta3/crypto/us/bars"
params = {
    "symbols": ticker,
    "start": "2023-01-01",
    "end": "2023-01-10",
    "timeframe": "1Day"
}
print(f"\nTesting Crypto {ticker}...")
r = requests.get(url, headers=headers, params=params)
print(f"Status: {r.status_code}")
if r.status_code != 200:
    print(f"Error: {r.text}")
