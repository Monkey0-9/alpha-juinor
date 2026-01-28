"""
tests/ingest/test_403_handling.py
Requirement: 403 blocks provider.
"""
import unittest
from unittest.mock import MagicMock, patch
from scripts.ingest_5y_batch import BatchIngestionAgent
from data.router.entitlement_router import router

class Test403Handling(unittest.TestCase):
    def setUp(self):
        router.blocked_providers = {}
        self.agent = BatchIngestionAgent("test_runs", "test_univ")

    @patch("scripts.ingest_5y_batch.BatchIngestionAgent._fetch_raw")
    def test_403_blocking(self, mock_fetch):
        # Simulate 403
        mock_fetch.side_effect = Exception("403 Forbidden: API Key Invalid")

        # Test logic directly
        df, actions, prov = self.agent._fetch_with_logic("AAPL", "yahoo", None, None)

        # Should be empty df
        self.assertTrue(df.empty)

        # Should be blocked in router
        # Note: The agent catches exception and calls router.block_provider

        is_blocked = router.is_blocked("yahoo", "AAPL")
        self.assertTrue(is_blocked, f"Provider NOT blocked. Mock call count: {mock_fetch.call_count}. Blocked state: {router.blocked_providers}")

if __name__ == "__main__":
    unittest.main()
