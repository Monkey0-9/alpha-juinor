
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ingestion.ingest_process import DataIngestionAgent

class TestDataIngestionAgent(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_router = MagicMock()

        # Patch DatabaseManager and DataRouter
        self.db_patcher = patch('data.ingestion.ingest_process.DatabaseManager', return_value=self.mock_db)
        self.router_patcher = patch('data.ingestion.ingest_process.DataRouter', return_value=self.mock_router)
        self.db_patcher.start()
        self.router_patcher.start()

        self.agent = DataIngestionAgent(tickers=['AAPL'])

    def tearDown(self):
        self.db_patcher.stop()
        self.router_patcher.stop()

    def test_calculate_quality_score_perfect(self):
        # Create a perfect dataframe
        dates = pd.date_range(start='2020-01-01', periods=1260, freq='B')
        df = pd.DataFrame({
            'Open': [100.0] * 1260,
            'High': [105.0] * 1260,
            'Low': [95.0] * 1260,
            'Close': [100.0] * 1260,
            'Volume': [1000] * 1260
        }, index=dates)

        quality = self.agent.calculate_quality_score(df)
        self.assertEqual(quality['score'], 1.0)
        self.assertEqual(quality['flags']['missing_dates_pct'], 0.0)

    def test_calculate_quality_score_missing_data(self):
        # Create a dataframe with half missing data (approx 630 rows)
        dates = pd.date_range(start='2020-01-01', periods=630, freq='B')
        df = pd.DataFrame({
            'Open': [100.0] * 630,
            'High': [105.0] * 630,
            'Low': [95.0] * 630,
            'Close': [100.0] * 630,
            'Volume': [1000] * 630
        }, index=dates)

        # Expected penalty: missing_dates_pct * 0.3
        # missing_dates_pct = (1260 - 630) / 1260 = 0.5
        # penalty = 0.5 * 0.3 = 0.15
        # score = 1.0 - 0.15 = 0.85

        quality = self.agent.calculate_quality_score(df)
        self.assertAlmostEqual(quality['score'], 0.85, places=2)

    def test_ingest_symbol_success(self):
        # Mock router classification
        self.mock_router._classify_ticker.return_value = 'US_EQUITY'
        self.mock_router.select_provider.return_value = 'alpaca'

        # Mock dataframe return
        dates = pd.date_range(start='2020-01-01', periods=1260, freq='B')
        df = pd.DataFrame({
            'Open': [100.0] * 1260,
            'High': [105.0] * 1260,
            'Low': [95.0] * 1260,
            'Close': [100.0] * 1260,
            'Volume': [1000] * 1260,
            'vwap': [100.0] * 1260,
            'trade_count': [100] * 1260
        }, index=dates)
        self.mock_router.get_price_history.return_value = df

        # Mock db transaction success
        self.mock_db.atomic_ingest.return_value = True

        self.agent.ingest_symbol('AAPL')

        # Verify db called
        self.mock_db.atomic_ingest.assert_called_once()
        self.assertEqual(self.agent.stats['successful'], 1)

    def test_ingest_symbol_no_provider(self):
        self.mock_router._classify_ticker.return_value = 'US_EQUITY'
        self.mock_router.select_provider.return_value = 'NO_VALID_PROVIDER'

        self.agent.ingest_symbol('AAPL')

        self.mock_db.atomic_ingest.assert_not_called()
        self.mock_db.log_ingestion_audit.assert_called()
        self.assertEqual(self.agent.stats['rejected'], 1)

if __name__ == '__main__':
    unittest.main()
