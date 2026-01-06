import pandas as pd
import numpy as np
import logging
from strategies.institutional_strategy import InstitutionalStrategy
from risk.engine import RiskManager
from data.utils.schema import ensure_dataframe

logging.basicConfig(level=logging.INFO)

def test_series_robustness():
    print("Running Series Robustness Test...")
    
    # 1. Create a Series (This used to crash the system)
    dates = pd.date_range("2025-01-01", periods=100)
    series_data = pd.Series(np.random.randn(100).cumsum() + 100, index=dates, name="Close")
    
    # 2. Test Strategy
    strat = InstitutionalStrategy()
    print("Testing Strategy with Series...")
    signals = strat.generate_signals(series_data)
    print(f"Strategy Signals generated: {signals.shape}")
    
    # 3. Test Risk Manager
    risk = RiskManager()
    print("Testing RiskManager with Series...")
    # Mocking target weights
    target_weights = {"Asset": 0.1}
    # This specifically failed in .dot(w_vector) if baskets_returns was a Series
    res = risk.check_pre_trade(target_weights, series_data, dates[-1])
    print(f"Risk Check Result: {res.ok}, Decision: {res.decision}")
    
    print("Test COMPLETE. No 'Series has no attribute columns' observed.")

if __name__ == "__main__":
    test_series_robustness()
