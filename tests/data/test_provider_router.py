"""
tests/data/test_provider_router.py
Requirement: Provider selection and entitlement blocking.
"""
import unittest
from unittest.mock import MagicMock, patch
from data.router.entitlement_router import router

class TestProviderRouter(unittest.TestCase):
    def setUp(self):
        # Reset runtime state
        router.blocked_providers = {}

    def test_select_stock(self):
        # Should pivot to alpaca/polygon/yahoo based on priority and capability
        # config says yahoo stocks=true, max_history=10000
        # alpaca stocks=true, max_history=730
        res = router.select_provider("AAPL", 500)
        self.assertNotEqual(res['provider'], "NONE")

    def test_select_fx_alpaca_limit(self):
        # Alpaca FX=False in my config update
        # Yahoo FX=True
        res = router.select_provider("EURUSD=X", 100)
        # Should be yahoo or polygon, NOT alpaca
        self.assertNotEqual(res['provider'], "alpaca")

    def test_history_limit(self):
        # Alpaca max 730
        res = router.select_provider("AAPL", 5000)
        # Should skip alpaca
        is_alpaca = res['provider'] == 'alpaca'
        self.assertFalse(is_alpaca, "Alpaca selected despite history limit")

    def test_runtime_block(self):
        router.block_provider("yahoo", "AAPL", "403 Forbidden")
        res = router.select_provider("AAPL", 100)
        # Yahoo is priority 1, but blocked. Should fallback to next if entitled.
        self.assertNotEqual(res['provider'], "yahoo")

if __name__ == "__main__":
    unittest.main()
