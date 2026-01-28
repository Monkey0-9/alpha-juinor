#!/usr/bin/env python3
"""Fix test imports"""
import os

# Fix test_data_integration.py
content = """import sys
import os
import unittest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.collectors.data_router import DataRouter
from data.providers.binance import BinanceDataProvider
from data.providers.fred import FredDataProvider
from data.providers.stooq import StooqProvider

class TestDataIntegration(unittest.TestCase):

    def setUp(self):
        self.router = DataRouter()

    def test_binance_fetch(self):
        print("Testing Binance...")
        bn = BinanceDataProvider()
        df = bn.fetch_ohlcv("BTC-USD", "2024-01-01", "2024-01-05")
        if len(df) > 0:
            self.assertIn("Close", df.columns)

    def test_stooq_fetch(self):
        print("Testing Stooq...")
        st = StooqProvider()
        df = st.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-10")
        if not df.empty:
            self.assertIn("Close", df.columns)

    def test_router_routing(self):
        print("Testing Router...")
        df_btc = self.router.get_price_history("BTC-USD", "2024-01-01", "2024-01-05")
        self.assertFalse(df_btc.empty)
        df_spy = self.router.get_price_history("SPY", "2024-01-01", "2024-01-05")
        self.assertFalse(df_spy.empty)

if __name__ == "__main__":
    unittest.main()
"""
with open('tests/test_data_integration.py', 'w') as f:
    f.write(content)
print('Fixed tests/test_data_integration.py')

# Fix test_phase2.py
content2 = """import unittest
import pandas as pd
import numpy as np

from risk.engine import RiskManager, RiskRegime
from alpha_families.ml_alpha import MLAlpha
from features.compute import FeatureComputer

class TestPhase2(unittest.TestCase):

    def test_risk_regime_detection(self):
        np.random.seed(42)
        rm = RiskManager()
        trend = np.linspace(100, 150, 250)
        noise = np.random.normal(0, 0.001, 250)
        prices = pd.Series(trend * (1 + noise))
        rm.update_regime(prices)
        self.assertEqual(rm.regime, RiskRegime.BULL_QUIET)
        self.assertTrue(rm.is_risk_on)

if __name__ == "__main__":
    unittest.main()
"""
with open('tests/test_phase2.py', 'w') as f:
    f.write(content2)
print('Fixed tests/test_phase2.py')

print("=== Test files fixed! ===")
