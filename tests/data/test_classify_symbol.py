"""
tests/data/test_classify_symbol.py
Requirement 13: Symbol classification for 100 edge cases (sampled).
"""
import unittest
from data.router.entitlement_router import router

class TestSymbolClassification(unittest.TestCase):
    def test_stocks(self):
        cases = ["AAPL", "MSFT", "GOOGL", "BRK.B", "T", "F"]
        for s in cases:
            self.assertEqual(router.classify_symbol(s), "stocks", f"Failed {s}")

    def test_fx(self):
        cases = ["EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X"]
        for s in cases:
            self.assertEqual(router.classify_symbol(s), "fx", f"Failed {s}")

    def test_crypto(self):
        cases = ["BTC-USD", "ETH-USD", "DOGE-USD", "USD-BTC"] # USD- prefix too per logic
        for s in cases:
            self.assertEqual(router.classify_symbol(s), "crypto", f"Failed {s}")

    def test_commodities(self):
        cases = ["GC=F", "CL=F", "SI=F", "NG=F"]
        for s in cases:
            self.assertEqual(router.classify_symbol(s), "commodities", f"Failed {s}")

    def test_edge_cases(self):
        # Mixed signals
        self.assertEqual(router.classify_symbol("BTC=F"), "commodities") # Suffix priority?
        # Logic: =X first (fx), then =F (commodities), then -USD (crypto).

if __name__ == "__main__":
    unittest.main()
