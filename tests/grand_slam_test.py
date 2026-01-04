# tests/grand_slam_test.py
import pytest
import pandas as pd
import numpy as np
import os
from datetime import datetime
from data.collectors.yahoo_collector import YahooDataProvider
from strategies.factory import StrategyFactory
from portfolio.allocator import InstitutionalAllocator
from risk.engine import RiskManager
from backtest.execution import RealisticExecutionHandler
from monitoring.alerts import AlertManager

def test_institutional_full_pipeline():
    """
    GRAND SLAM TEST: Verifies all institutional modules work in harmony.
    """
    print("\n[Phase 1] Initializing Modular Components...")
    risk = RiskManager(max_leverage=1.0, target_vol_limit=0.15)
    allocator = InstitutionalAllocator(risk_manager=risk, max_leverage=1.0)
    alerts = AlertManager()
    
    config = {
        "type": "institutional",
        "tickers": ["SPY", "TLT"],
        "use_ml": False,
        "alpha_window": 20
    }
    strategy = StrategyFactory.create_strategy(config)
    
    # Mock Market Data
    dates = pd.date_range("2023-01-01", periods=100)
    data = {}
    for tk in ["SPY", "TLT"]:
        prices = np.linspace(100, 110, 100) + np.random.normal(0, 1, 100)
        data[(tk, "Open")] = prices
        data[(tk, "High")] = prices + 0.5
        data[(tk, "Low")] = prices - 0.5
        data[(tk, "Close")] = prices
        data[(tk, "Volume")] = [1_000_000] * 100
        
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    print("[Phase 2] Running Signal Generation...")
    signals_df = strategy.generate_signals(df)
    signals = signals_df.iloc[-1].to_dict()
    assert "SPY" in signals
    assert "TLT" in signals
    print(f"   Signals: {signals}")

    print("[Phase 3] Running Kelly Allocation...")
    prices_map = {tk: df[tk]["Close"] for tk in ["SPY", "TLT"]}
    vols_map = {tk: df[tk]["Volume"] for tk in ["SPY", "TLT"]}
    
    class MockPortfolio:
        def __init__(self):
            self.total_equity = 1_000_000.0
            self.positions = {}
    
    alloc_res = allocator.allocate(
        signals=signals,
        prices=prices_map,
        volumes=vols_map,
        current_portfolio=MockPortfolio(),
        timestamp=dates[-1],
        method="kelly"
    )
    
    print(f"   Target Weights: {alloc_res.target_weights}")
    assert len(alloc_res.orders) > 0
    
    print("[Phase 4] Verifying Alerting...")
    alerts.alert("Grand Slam Integration Test - Phase 4 OK", level="SUCCESS")
    
    print("\n✅ Institutional Grand Slam Successful!")

if __name__ == "__main__":
    test_institutional_full_pipeline()
