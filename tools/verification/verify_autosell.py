# verify_autosell.py
import pandas as pd
import numpy as np
import logging
from risk.engine import RiskManager, RiskRegime
from portfolio.allocator import InstitutionalAllocator
from backtest.execution import Order

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoSellVerify")

class MockPortfolio:
    def __init__(self, equity, positions):
        self.total_equity = equity
        self.positions = positions

def verify():
    # 1. Setup Risk Manager and Allocator
    rm = RiskManager(initial_capital=100000.0)
    allocator = InstitutionalAllocator(rm)
    
    # 2. Mock Data
    tickers = ["AAPL", "BTC-USD"]
    ts = pd.Timestamp.now(tz='UTC')
    
    # Prices: AAPL=150, BTC=50000
    prices = {
        "AAPL": pd.Series([140, 145, 150], index=[ts-pd.Timedelta(days=2), ts-pd.Timedelta(days=1), ts]),
        "BTC-USD": pd.Series([48000, 49000, 50000], index=[ts-pd.Timedelta(days=2), ts-pd.Timedelta(days=1), ts])
    }
    volumes = {
        "AAPL": pd.Series([1000000, 1000000, 1000000], index=[ts-pd.Timedelta(days=2), ts-pd.Timedelta(days=1), ts]),
        "BTC-USD": pd.Series([500, 500, 500], index=[ts-pd.Timedelta(days=2), ts-pd.Timedelta(days=1), ts])
    }
    
    # Current Positions: 100 AAPL ($15k), 0.1 BTC ($5k). Total = $20k
    current_positions = {"AAPL": 100, "BTC-USD": 0.1}
    portfolio = MockPortfolio(100000.0, current_positions)
    
    # Case A: Signal Deterioration for AAPL (Target = 0)
    print("\n--- CASE A: SIGNAL DETERIORATION (AAPL TARGET -> 0) ---")
    signals = {"AAPL": 0.5, "BTC-USD": 0.6} # 0.5 is neutral/exit
    res = allocator.allocate(signals, prices, volumes, portfolio, ts)
    for o in res.orders:
        if o.ticker == "AAPL":
             print(f"Order: {o.ticker} | Qty: {o.quantity} | Reason: {o.reason}")
    
    # Case B: Risk Breach (Drawdown)
    print("\n--- CASE B: RISK BREACH (DRAWDOWN) ---")
    rm._max_equity = 120000.0 # Huge drawdown from 120k to 100k
    signals = {"AAPL": 0.8, "BTC-USD": 0.8}
    res = allocator.allocate(signals, prices, volumes, portfolio, ts)
    for o in res.orders:
        print(f"Order: {o.ticker} | Qty: {o.quantity} | Reason: {o.reason} | Metrics: {o.risk_metric_triggered}")

    # Case C: Emergency Kill
    print("\n--- CASE C: EMERGENCY KILL ---")
    rm.manual_emergency_flag = True
    rm.check_circuit_breaker(100000.0, pd.Series([0,0,0]))
    res = allocator.allocate(signals, prices, volumes, portfolio, ts)
    for o in res.orders:
        print(f"Order: {o.ticker} | Qty: {o.quantity} | Reason: {o.reason}")

    # Case D: Safety Shield (Never Sell)
    print("\n--- CASE D: SAFETY SHIELD (NEVER SELL AAPL) ---")
    rm.manual_emergency_flag = False
    from risk.engine import RiskDecision
    rm.state = RiskDecision.ALLOW
    rm.never_sell_assets = ["AAPL"]
    signals_d = {"AAPL": 0.2, "BTC-USD": 0.2} # Want to sell everything
    res_d = allocator.allocate(signals_d, prices, volumes, portfolio, ts)
    for o in res_d.orders:
        print(f"Order: {o.ticker} | Qty: {o.quantity} | Reason: {o.reason}")

    for o in res.orders:
        print(f"Order: {o.ticker} | Qty: {o.quantity} | Reason: {o.reason}")

    # Case E: Hysteresis - No Entry on Weak Signal (0.6)
    print("\n--- CASE E: HYSTERESIS (NO ENTRY ON WEAK SIGNAL 0.6) ---")
    signals = {"AAPL": 0.6, "BTC-USD": 0.5} 
    current_positions_e = {"AAPL": 0, "BTC-USD": 0}
    portfolio_e = MockPortfolio(100000.0, current_positions_e)
    res_e = allocator.allocate(signals, prices, volumes, portfolio_e, ts)
    print(f"Orders count: {len(res_e.orders)}")

    # Case F: Hysteresis - Exit Buffer (0.6)
    print("\n--- CASE F: HYSTERESIS (STAY IN ON 0.6 IF ALREADY POS) ---")
    current_positions_f = {"AAPL": 100}
    portfolio_f = MockPortfolio(100000.0, current_positions_f)
    res_f = allocator.allocate(signals, prices, volumes, portfolio_f, ts)
    print(f"Orders for AAPL: {[o.quantity for o in res_f.orders if o.ticker == 'AAPL']}")

    # Case G: Adaptive Stop-Loss
    print("\n--- CASE G: ADAPTIVE STOP-LOSS ---")
    allocator.entry_prices["AAPL"] = 200.0 # Entry at 200, current=150
    signals_g = {"AAPL": 0.9, "BTC-USD": 0.5} 
    res_g = allocator.allocate(signals_g, prices, volumes, portfolio_f, ts)
    for o in res_g.orders:
        if o.ticker == "AAPL":
            print(f"Order: {o.ticker} | Qty: {o.quantity} | Reason: {o.reason} | Metric: {o.risk_metric_triggered}")

if __name__ == "__main__":
    verify()
