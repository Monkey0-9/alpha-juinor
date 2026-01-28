
import os
import requests
from dotenv import load_dotenv

def test_alpaca_connection():
    load_dotenv(override=True)
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_API_URL", "https://paper-api.alpaca.markets")

    print(f"Testing connection to: {base_url}")
    print(f"API Key: {api_key[:5]}...")

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }

    try:
        response = requests.get(f"{base_url}/v2/account", headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("SUCCESS! Connected to Alpaca.")
            print(f"Status: {data.get('status')}")
            print(f"Equity: ${data.get('equity')}")
            print(f"Currency: {data.get('currency')}")
        else:
            print(f"FAILED! Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_alpaca_connection()
