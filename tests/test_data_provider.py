import unittest
import pandas as pd
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from data.provider import YahooDataProvider
from data.storage import DataStore

class TestDataLayer(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = Path("tests/temp_data")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()
        
        # Create mock data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="B")
        data = {
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low": [95.0] * 10,
            "Close": [102.0] * 10,
            "Volume": [1000] * 10
        }
        self.mock_data = pd.DataFrame(data, index=dates)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_yahoo_provider_fetch(self):
        provider = YahooDataProvider()
        
        with patch("yfinance.download") as mock_yf:
            mock_yf.return_value = self.mock_data
            
            df = provider.fetch_ohlcv("TEST", start_date="2023-01-01")
            
            self.assertFalse(df.empty)
            self.assertTrue("Close" in df.columns)
            self.assertEqual(len(df), 10)

    def test_data_store_save_load(self):
        store = DataStore(data_dir=str(self.temp_dir))
        ticker = "TEST_TICKER"
        
        # Save
        store.save(ticker, self.mock_data)
        self.assertTrue(store.exists(ticker))
        
        # Load
        loaded_df = store.load(ticker)
        # CSV loses frequency information, so we ignore it
        pd.testing.assert_frame_equal(self.mock_data, loaded_df, check_freq=False)

    def test_data_store_missing_file(self):
        store = DataStore(data_dir=str(self.temp_dir))
        df = store.load("NON_EXISTENT")
        self.assertTrue(df.empty)

if __name__ == "__main__":
    unittest.main()
