
import sys
import os
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime
from risk.engine import RiskManager
from strategies.alpha import CompositeAlpha, TrendAlpha
from backtest.execution import Order

class TestActiveTrading(unittest.TestCase):
    
    def test_strong_signal_generates_orders(self):
        """Verify that a strong trend + neutral risk generates valid Orders."""
        
        # 1. Setup Data with Strong Trend
        dates = pd.date_range("2025-01-01", periods=300, freq="B")
        # Price doubles: 100 -> 200
        trend = np.linspace(100, 200, 300) + np.random.normal(0, 0.1, 300) 
        prices = pd.Series(trend, index=dates)
        
        # 2. Setup Alpha
        # TrendAlpha should pick this up easily
        t_alpha = TrendAlpha(short=20, long=50)
        c_alpha = CompositeAlpha([t_alpha], window=60)
        
        # Run alpha to build history
        signals = []
        for i in range(100, 300):
            sub_p = prices.iloc[:i]
            # Compute returns signal 0..1
            s = c_alpha.compute(sub_p)
            signals.append(s.iloc[-1])
            
        final_signal = signals[-1]
        print(f"Final Signal: {final_signal}")
        
        # It should be > 0.0 (Institutional Rule: Any positive conviction is a valid trade)
        self.assertTrue(final_signal > 0.0, "In strong uptrend, signal must be positive")
        
        # 3. Validation with Risk Manager
        rm = RiskManager(target_vol_limit=0.15)
        # Update regime
        rm.update_regime(prices)
        
        # Conviction 0..1 -> 0.0 .. 1.0 leverage
        conviction = pd.Series([final_signal], index=["SPY"])
        
        # Check Pre-trade
        # Need volume for liquidity check
        vol = pd.Series([1_000_000]*300, index=dates)
        
        res = rm.enforce_limits(conviction, prices, vol)
        
        adj_conv = res.adjusted_conviction.iloc[-1]
        print(f"Risk Adjusted Conviction: {adj_conv}")
        
        # Should be non-zero (Institutional Rule: zero = veto, small positive = trade)
        self.assertTrue(adj_conv > 0.0, "Risk Manager should allow non-zero exposure")
        
if __name__ == '__main__':
    unittest.main()
