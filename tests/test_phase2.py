
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from risk.engine import RiskManager, RiskRegime
from strategies.alpha import CompositeAlpha, TrendAlpha
from strategies.ml_alpha import MLAlpha
from strategies.features import FeatureEngineer

class TestPhase2(unittest.TestCase):
    
    def test_risk_regime_detection(self):
        """Verify RiskManager detects Bull/Bear and Volatility regimes."""
        rm = RiskManager()
        
        # 1. Create Bull Quiet Scenario (Price > MA, Low Vol)
        # MA200 needs 200 pts. We'll generate 250.
        # Linear uptrend: 100 -> 150
        trend = np.linspace(100, 150, 250)
        # Add very small noise (Low Vol)
        noise = np.random.normal(0, 0.001, 250)
        prices = pd.Series(trend * (1 + noise))
        
        rm.update_regime(prices)
        
        self.assertEqual(rm.regime, RiskRegime.BULL_QUIET, "Should be BULL_QUIET")
        self.assertTrue(rm.is_risk_on)
        
        # 2. Create Bear Crisis Scenario (Price < MA, High Vol)
        # Downtrend: 100 -> 50
        downtrend = np.linspace(100, 50, 250)
        # High noise
        noise_high = np.random.normal(0, 0.02, 250) # 2% daily -> 32% annual
        prices_bad = pd.Series(downtrend * (1 + noise_high))
        
        rm = RiskManager() # Reset
        rm.update_regime(prices_bad)
        
        # Vol percentile needs history. We simulate feed.
        # But our update_regime logic calculates vol from passed history tail.
        # History > 50 points triggers percentile logic.
        # Wait, rm._vol_history needs to be populated over time in real app.
        # In test, we might only get 1 data point of vol from update_regime call?
        # update_regime: "self._vol_history.append(current_vol)"
        # If we call it once, len is 1. Percentile is 0.5.
        # So we need to call it multiple times or manually populate?
        # Let's populate _vol_history to simulate high vol context
        rm._vol_history = [0.01] * 100 # Low vol history
        # Now we feed high vol regime.
        
        rm.update_regime(prices_bad)
        # Current vol will be high (~30%). Percentile should be 1.0 (High).
        # Price (approx 50) < MA200 (approx 75). -> Downtrend.
        # Should be BEAR_CRISIS.
        
        self.assertEqual(rm.regime, RiskRegime.BEAR_CRISIS, "Should be BEAR_CRISIS")
        self.assertFalse(rm.is_risk_on)

    def test_composite_dynamic_weighting(self):
        """Verify CompositeAlpha adjusts weights based on IC."""
        # 2 Alphas:
        # A1: Perfect correlation with future return
        # A2: Negative/Zero correlation
        
        class MockAlpha:
            def __init__(self, val_series):
                self.vals = val_series
            def compute(self, prices):
                # Return pre-defined signal for length of prices
                l = len(prices)
                return self.vals.iloc[:l]

        # Scenario: Prices go UP 1% every step.
        prices = pd.Series(np.linspace(100, 110, 100)) # 100 steps
        
        # Alpha 1: Predicts 1.0 everywhere (Correct, since price goes up)
        a1_sigs = pd.Series([1.0]*100)
        # Alpha 2: Predicts 0.0 everywhere (Neutral/Wrong?)
        # Or let's make it predict -1.0 (Wrong)
        a2_sigs = pd.Series([-1.0]*100)
        
        comp = CompositeAlpha([MockAlpha(a1_sigs), MockAlpha(a2_sigs)])
        
        # Feed data step by step to build history
        for i in range(10, 60):
            sub_p = prices.iloc[:i]
            comp.compute(sub_p)
            
        # After ~50 steps:
        # A1 signal (1.0) vs Return (+). IC should be NaN (const signal)?
        # Ah, if signal is constant 1.0, std_dev is 0. Correlation is NaN.
        # weight logic handles std<1e-8 -> IC=0.
        
        # We need VARIANCE in signal to have correlation.
        # Let's make A1 signal fluctuate WITH returns.
        # Returns are roughly constant positive. So we need constant positive signal?
        # No, correlation requires covariance.
        # If returns are constant, correlation is defined? No, std(target)=0 -> corr=NaN.
        
        # Better Test Data: Sine Wave Prices
        t = np.linspace(0, 10, 100)
        p_vals = 100 + np.sin(t)
        prices = pd.Series(p_vals)
        # Returns roughly follow cos(t)
        
        # A1: Follows cos(t) -> High IC
        a1_vals = np.cos(t)
        # A2: Follows -cos(t) -> Negative IC
        a2_vals = -np.cos(t)
        
        comp = CompositeAlpha([MockAlpha(pd.Series(a1_vals)), MockAlpha(pd.Series(a2_vals))])
        
        # Run
        for i in range(20, 80):
            comp.compute(prices.iloc[:i])
            
        # Check weights. A1 should have much higher weight than A2.
        w = comp.weights
        print(f"Weights: {w}")
        self.assertTrue(w[0] > 0.8, f"A1 should dominate weights (got {w[0]})")
        self.assertTrue(w[1] < 0.2, f"A2 should have low weight (got {w[1]})")

if __name__ == '__main__':
    unittest.main()
