"""
tests/data/test_entitlement_router.py
"""

import unittest
import os
import json
from data.router.entitlement_router import EntitlementRouter, router

class TestEntitlementRouter(unittest.TestCase):

    def setUp(self):
        # Reset router state
        if os.path.exists(router.blocked_file):
            os.remove(router.blocked_file)
        router._init() # Reload

    def test_classification(self):
        self.assertEqual(router.classify_symbol("EURUSD=X"), "fx")
        self.assertEqual(router.classify_symbol("BTC-USD"), "crypto")
        self.assertEqual(router.classify_symbol("ES=F"), "commodities")
        self.assertEqual(router.classify_symbol("AAPL"), "stocks")

    def test_selection_logic(self):
        # Yahoo supports everything, max 5000
        # Alpaca supports stocks/crypto, max 730

        # Stock: Long history -> Yahoo
        res = router.select_provider("AAPL", 4000)
        self.assertEqual(res["provider"], "yahoo")

        # Stock: Short history -> Yahoo (priority 1) or Polygon (priority 2) or Alpaca (priority 3)
        # Config priority: yahoo, polygon, alpaca
        res = router.select_provider("AAPL", 100)
        self.assertEqual(res["provider"], "yahoo")

        # Crypto: Alpaca supports, Yahoo supports
        # Priority order
        res = router.select_provider("BTC-USD", 100)
        self.assertEqual(res["provider"], "yahoo")

    def test_blocking(self):
        # Manual block
        router.block_provider("yahoo", "AAPL", "403_FORBIDDEN")

        # Should skip yahoo
        res = router.select_provider("AAPL", 100)
        self.assertNotEqual(res["provider"], "yahoo")
        # Should likely pick polygon or alpaca if entitled

    def test_capability_constraints(self):
        # Alpaca has max_history_days = 730
        # Determine if EntitlementRouter respects it
        # If we force Alpaca to be top priority temporarily
        original_priority = router.provider_priority
        router.provider_priority = ["alpaca", "yahoo"]

        # 800 days > 730 -> Should skip Alpaca and go Yahoo
        res = router.select_provider("TSLA", 800)
        self.assertEqual(res["provider"], "yahoo")

        # Restore
        router.provider_priority = original_priority

if __name__ == "__main__":
    unittest.main()
