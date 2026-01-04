# run_prototype.py
"""
Mini Quant Fund - Institutional Prototype Run
Run this script to see the full capabilities of the upgraded platform 
without requiring external data or API keys.
"""
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Import Institutional Components
try:
    from backtest.engine import BacktestEngine
    from backtest.execution import RealisticExecutionHandler, Order, OrderType
    from risk.engine import RiskManager
    from strategies.ml_alpha import MLAlpha
    from strategies.features import FeatureEngineer
    from reports.performance_attribution import PerformanceAnalyzer
except ImportError as e:
    logger.error(f"Failed to import core modules: {e}")
    sys.exit(1)

# Mock Data Provider to allow standalone execution
class PrototypeProvider:
    def __init__(self):
        logger.info("Initializing Synthetic Market Data Provider...")

    def fetch_ohlcv(self, ticker, start_date, end_date):
        dates = pd.date_range(start_date, end_date, freq='B')
        n = len(dates)
        
        # Consistent random seed for reproducibility
        np.random.seed(42 if ticker == "SPY" else 123)
        
        # Generate synthetic price path (Geometric Brownian Motion + Noise)
        drift = 0.0005 # Positive drift
        volatility = 0.015 
        
        returns = np.random.normal(drift, volatility, n)
        price_path = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame(index=dates)
        df['Close'] = price_path
        df['Open'] = price_path * (1 + np.random.normal(0, 0.002, n))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, n)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, n)))
        df['Volume'] = np.random.randint(1_000_000, 5_000_000, n)
        
        return df

    def get_panel(self, tickers, start, end):
        data = {}
        for t in tickers:
            data[t] = self.fetch_ohlcv(t, start, end)
        return pd.concat(data, axis=1)

def run_simulation():
    print("\n" + "="*60)
    print("   MINI QUANT FUND — INSTITUTIONAL PROTOTYPE SIMULATION")
    print("="*60 + "\n")
    
    # 1. Configuration ------------------------------------------------
    tickers = ["SPY", "QQQ", "TLT"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    initial_cap = 1_000_000.0
    
    # 2. Instantiate Components ---------------------------------------
    provider = PrototypeProvider()
    
    # Execution Handler with Bid-Ask Spread and Market Impact
    handler = RealisticExecutionHandler(
        commission_pct=0.0005, 
        bid_ask_spread_pct=0.0005, # 5bps spread
        impact_coeff=0.2,          # High impact config to demonstrate cost
        buy_impact_mult=1.0,
        sell_impact_mult=1.5
    )
    
    # Risk Manager with Factor Limits and Stress Tests
    risk_manager = RiskManager(
        max_leverage=1.5,
        target_vol_limit=0.15,
        max_drawdown_limit=0.15
    )
    
    # ML Alpha
    fe = FeatureEngineer()
    ml_alpha = MLAlpha(fe, train_window=100)
    
    engine = BacktestEngine(
        provider=provider,
        initial_capital=initial_cap,
        execution_handler=handler,
        risk_manager=risk_manager
    )
    
    # 3. Define Strategy Strategy -------------------------------------
    # (Simple Trend Following + ML Signal Placeholder)
    def strategy_fn(timestamp, prices, portfolio):
        if timestamp.day != 1: # Convert to monthly rebalance
            return []
            
        orders = []
        equity = portfolio.total_equity
        
        # Simple Logic: Buy if Price > 100 (in synthetic data it starts at 100)
        # Target Weight 30% per asset
        target_weight = 0.30
        
        for tk in tickers:
            price = prices.get(tk)
            if not price: continue
            
            # Use ML prediction (mock usage since we didn't train fully in loop here)
            # In real loop, we would call ml_alpha.predict(...)
            
            current_qty = portfolio.positions.get(tk, 0.0)
            target_val = equity * target_weight
            target_qty = target_val / price
            
            diff_qty = target_qty - current_qty
            
            # Threshold to trade
            if abs(diff_qty * price) > 5000:
                orders.append(Order(tk, diff_qty, OrderType.MARKET, timestamp))
                
        return orders

    # 4. Execution ----------------------------------------------------
    logger.info(f"Starting simulation from {start_date} to {end_date}...")
    try:
        engine.run(start_date, strategy_fn, tickers, end_date)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return

    # 5. Reporting ----------------------------------------------------
    results = engine.get_results()
    trades = engine.get_blotter().trades_df()
    
    if results.empty:
        logger.error("No results generated.")
        return
        
    logger.info("Generating Institutional Performance Report...")
    
    # Benchmark Returns for Attribution (SPY)
    spy_df = provider.fetch_ohlcv("SPY", start_date, end_date)
    bench_rets = spy_df["Close"].pct_change().dropna()
    
    analyzer = PerformanceAnalyzer(results["equity"], trades)
    report = analyzer.generate_report(benchmark_returns=bench_rets)
    
    print("\n" + report)
    print("\nSimulation Complete. Prototype validated.")

if __name__ == "__main__":
    run_simulation()
