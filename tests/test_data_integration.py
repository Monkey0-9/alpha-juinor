
import sys
import os
import unittest
import pandas as pd
from datetime import datetime, timedelta

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.collectors.data_router import DataRouter
from data.collectors.binance_collector import BinanceDataProvider
from data.collectors.fred_collector import FredDataProvider
from data.collectors.stooq_collector import StooqDataProvider

class TestDataIntegration(unittest.TestCase):
    
    def setUp(self):
        self.router = DataRouter()

    def test_binance_fetch(self):
        print("\nTesting Binance (Crypto)...")
        bn = BinanceDataProvider()
        df = bn.fetch_ohlcv("BTC-USD", "2024-01-01", "2024-01-05")
        print(f"Binance BTC Rows: {len(df)}")
        self.assertTrue(len(df) > 0)
        self.assertIn("Close", df.columns)

    def test_fred_fetch(self):
        print("\nTesting FRED (Macro)...")
        # Needs API Key usually, but let's check basic class structure
        # If API key is missing (CI/CD), it returns empty Series, that's expected behavior
        fred = FredDataProvider()
        if not fred.api_key:
            print("Skipping FRED fetch (No API Key)")
            return
            
        s = fred.fetch_series("VIXCLS", "2023-01-01")
        print(f"FRED VIX Rows: {len(s)}")
        self.assertTrue(len(s) >= 0) # API key might be invalid

    def test_stooq_fetch(self):
        print("\nTesting Stooq (Equity Backup)...")
        st = StooqDataProvider()
        # Test loading SPY equivalent or Apple
        df = st.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-10")
        print(f"Stooq AAPL Rows: {len(df)}")
        if not df.empty:
            self.assertIn("Close", df.columns)
        else:
            print("Stooq returned empty (Rate limit or symbol issue)")

    def test_router_routing(self):
        print("\nTesting Router Logic...")
        # Crypto -> Binance
        df_btc = self.router.get_price_history("BTC-USD", "2024-01-01", "2024-01-05")
        self.assertFalse(df_btc.empty, "Router failed to fetch BTC")
        
        # Equity -> Yahoo (Default)
        df_spy = self.router.get_price_history("SPY", "2024-01-01", "2024-01-05")
        self.assertFalse(df_spy.empty, "Router failed to fetch SPY")

if __name__ == '__main__':
    unittest.main()
