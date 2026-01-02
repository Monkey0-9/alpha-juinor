import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.provider import YahooDataProvider
from data.storage import DataStore

ASSETS = ["SPY", "QQQ", "TLT", "GLD", "IWM", "EEM", "AGG", "LQD"]
START_DATE = "2010-01-01"

def run_ingestion():
    print(f"Starting data ingestion for {len(ASSETS)} assets...")
    
    provider = YahooDataProvider()
    store = DataStore()
    
    for ticker in ASSETS:
        try:
            df = provider.fetch_ohlcv(ticker, start_date=START_DATE)
            if not df.empty:
                store.save(ticker, df)
        except Exception as e:
            print(f"   [Error] Failed to ingest {ticker}: {e}")
            
    print("Ingestion complete.")

if __name__ == "__main__":
    run_ingestion()
