# verify_decision_layer.py
import pandas as pd
import numpy as np
import logging
from risk.engine import RiskManager, RiskDecision
from portfolio.allocator import InstitutionalAllocator
from backtest.execution import Order

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DecisionLayerVerify")

class MockPortfolio:
    def __init__(self, equity, positions):
        self.total_equity = equity
        self.positions = positions

def verify():
    rm = RiskManager(initial_capital=100000.0)
    allocator = InstitutionalAllocator(rm)
    ts = pd.Timestamp.now(tz='UTC')
    
    # Prices: 15 points
    dates = [ts - pd.Timedelta(days=i) for i in range(15)]
    dates.reverse()
    # AAPL: 190 -> 200. Slow uptrend.
    # BTC: 50000 -> 52000.
    prices = {
        "AAPL": pd.Series([190 + i*0.7 for i in range(15)], index=dates),
        "BTC-USD": pd.Series([50000 + i*150 for i in range(15)], index=dates)
    }
    volumes = {k: pd.Series([1000000]*15, index=prices[k].index) for k in prices}

    portfolio_empty = MockPortfolio(100000.0, {})
    
    # 1. Hysteresis Entry (0.6 is not enough)
    print("\n[TEST 1] Hysteresis Entry (Sig=0.6, Threshold=0.65)")
    res = allocator.allocate({"AAPL": 0.6}, prices, volumes, portfolio_empty, ts)
    print(f"Orders: {len(res.orders)} (Expected: 0)")

    # 2. Hysteresis Entry (0.7 is enough)
    print("\n[TEST 2] Hysteresis Entry (Sig=0.7, Threshold=0.65)")
    res = allocator.allocate({"AAPL": 0.7}, prices, volumes, portfolio_empty, ts)
    print(f"Orders: {len(res.orders)} (Expected: 1)")
    
    # 3. Trend Confirmation (Signal must rise)
    print("\n[TEST 3] Trend Confirmation (Sig stays 0.7)")
    # allocator.last_signals["AAPL"] is now 0.7 from previous call
    res = allocator.allocate({"AAPL": 0.7}, prices, volumes, portfolio_empty, ts)
    print(f"Orders: {len(res.orders)} (Expected: 0 because sig did not rise)")

    # 4. Hysteresis Exit (0.6 is fine if held)
    print("\n[TEST 4] Hysteresis Exit (Hold at 0.6, Exit at 0.55)")
    portfolio_filled = MockPortfolio(100000.0, {"AAPL": 100})
    res = allocator.allocate({"AAPL": 0.6}, prices, volumes, portfolio_filled, ts)
    print(f"Orders for AAPL: {[o.quantity for o in res.orders if o.ticker == 'AAPL']} (Expected: 0 or minor rebalance)")

    # 5. Adaptive Stop
    print("\n[TEST 5] Adaptive Stop Trigger")
    allocator.entry_prices["AAPL"] = 300.0 # Huge drop from 300 to 200
    res = allocator.allocate({"AAPL": 0.9}, prices, volumes, portfolio_filled, ts)
    for o in res.orders:
        if o.ticker == "AAPL":
            print(f"Order: {o.ticker} Qty: {o.quantity} Reason: {o.reason} Metric: {o.risk_metric_triggered}")
            
    # 6. Cooldown (Intelligent Re-entry)
    print("\n[TEST 6] Cooldown Guard (Sig=0.9 but just stopped)")
    res = allocator.allocate({"AAPL": 0.9}, prices, volumes, portfolio_empty, ts)
    print(f"Orders: {len(res.orders)} (Expected: 0 due to cooldown)")

if __name__ == "__main__":
    verify()
