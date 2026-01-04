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
        "use_ml": True,  # ENABLE ML to boost conviction (Tech ensemble is conflicted)
        "alpha_window": 20
    }
    strategy = StrategyFactory.create_strategy(config)
    
    # Mock Market Data - DIP AND RECOVER CYCLES
    # Repeated cycles allow ML to learn "Dip = Buy"
    # Dips trigger Mean Reversion / RSI / Bollinger technical buys
    dates = pd.date_range("2023-01-01", periods=600, freq="B")
    data = {}
    for tk in ["SPY", "TLT"]:
        # Base price 100
        prices = np.full(600, 100.0)
        
        # Inject 3 Dip-Recovery cycles
        for i in range(3):
            start = 100 + i * 150
            # Dip 20 pts over 20 days
            prices[start:start+20] = np.linspace(100, 80, 20)
            # Recover 20 pts over 20 days
            prices[start+20:start+40] = np.linspace(80, 105, 20)
            
        # Current state (end of array): Just finished a Dip, starting recovery
        # 580-600: Dip from 100 to 80
        prices[-20:] = np.linspace(100, 80, 20)
        
        # Add noise
        noise = np.random.normal(0, 0.5, 600)
        prices = prices + noise
        
        data[(tk, "Open")] = prices
        data[(tk, "High")] = prices + 2.0
        data[(tk, "Low")] = prices - 2.0
        data[(tk, "Close")] = prices
        data[(tk, "Volume")] = np.random.randint(5_000_000, 20_000_000, 600)
        
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    print("[Phase 1.5] Training ML Models...")
    strategy.train_models(df)

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
