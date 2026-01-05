import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from data.providers.yahoo import YahooDataProvider
from data.providers.binance import BinanceDataProvider
from data.providers.coingecko import CoinGeckoDataProvider
from data.providers.polygon import PolygonDataProvider

class TestDataProviders(unittest.TestCase):
    """
    Integration tests for all data providers.
    """

    def setUp(self):
        # Create mock OHLCV data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        self.mock_ohlcv = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [105.0] * 10,
            "Low": [95.0] * 10,
            "Close": [102.0] * 10,
            "Volume": [1000] * 10
        }, index=dates)

    def test_yahoo_provider(self):
        provider = YahooDataProvider(enable_cache=False)  # Disable cache for testing

        with patch("yfinance.download") as mock_yf:
            mock_yf.return_value = self.mock_ohlcv

            df = provider.fetch_ohlcv("AAPL", start_date="2023-01-01", end_date="2023-01-10")

            self.assertFalse(df.empty)
            self.assertEqual(len(df), 10)
            self.assertTrue(all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]))

    def test_binance_provider(self):
        provider = BinanceDataProvider()

        mock_response = MagicMock()
        mock_response.json.return_value = [
            [1640995200000, "100.0", "105.0", "95.0", "102.0", "1000.0", 1640998800000, "1000.0", 100, "500.0", "500.0", "0"]
        ] * 10
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            df = provider.fetch_ohlcv("BTCUSDT", start_date="2023-01-01", end_date="2023-01-10")

            self.assertFalse(df.empty)
            self.assertTrue(all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]))

    def test_coingecko_provider(self):
        provider = CoinGeckoDataProvider(enable_cache=False)

        # Mock CoinGecko OHLC response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [1640995200000, 100.0, 105.0, 95.0, 102.0]  # [timestamp, open, high, low, close]
        ] * 10
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            df = provider.fetch_ohlcv("BTC", start_date="2023-01-01", end_date="2023-01-10")

            self.assertFalse(df.empty)
            self.assertTrue(all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]))

    def test_polygon_provider(self):
        provider = PolygonDataProvider(enable_cache=False)

        # Mock Polygon aggregates response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"t": 1640995200000, "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000.0}
            ] * 10
        }
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            df = provider.fetch_ohlcv("AAPL", start_date="2023-01-01", end_date="2023-01-10")

            self.assertFalse(df.empty)
            self.assertTrue(all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]))

    def test_provider_panel_methods(self):
        """Test get_panel method for each provider."""
        providers = [
            YahooDataProvider(enable_cache=False),
            CoinGeckoDataProvider(enable_cache=False),
            PolygonDataProvider(enable_cache=False)
        ]

        for provider in providers:
            with self.subTest(provider=provider.__class__.__name__):
                # Mock the fetch_ohlcv method
                with patch.object(provider, 'fetch_ohlcv', return_value=self.mock_ohlcv):
                    panel = provider.get_panel(["TEST"], start_date="2023-01-01", end_date="2023-01-10")

                    self.assertFalse(panel.empty)
                    self.assertIsInstance(panel.columns, pd.MultiIndex)

if __name__ == "__main__":
    unittest.main()
