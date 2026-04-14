
import os
import logging
from dotenv import load_dotenv
from mini_quant_fund.data.collectors.alpaca_collector import AlpacaDataProvider
from mini_quant_fund.brokers.alpaca_broker import AlpacaExecutionHandler

logging.basicConfig(level=logging.INFO)
load_dotenv()

def verify():
    print("Testing Alpaca Connection...")
    try:
        provider = AlpacaDataProvider()
        price = provider.get_latest_quote("SPY")
        print(f"Latest SPY Price: {price}")
        
        handler = AlpacaExecutionHandler(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_SECRET_KEY")
        )
        acc = handler.get_account()
        print(f"Account Status: {acc.get('status')} | Equity: {acc.get('equity')}")
        
    except Exception as e:
        print(f"Verification FAILED: {e}")

if __name__ == "__main__":
    verify()
