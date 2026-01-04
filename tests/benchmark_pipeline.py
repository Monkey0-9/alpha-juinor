
import sys
import os
import time
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.institutional_strategy import InstitutionalStrategy
from configs.config_manager import ConfigManager

def run_benchmark():
    print("🚀 Starting High-Frequency Benchmark...")
    
    # 1. Setup Strategy
    config = {
        'tickers': ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT'], # 5 Ticker Universe
        'features': {
            'use_regime_detection': True,
            'use_wyckoff_filter': True,
            'use_auction_market_confidence': True,
            'use_market_profile_value_area': True,
            'use_gann_time_filter': True,
            'use_vpin_filter': True
        }
    }
    strategy = InstitutionalStrategy(config)
    print(f"✅ Strategy Initialized with {len(config['tickers'])} tickers.")
    
    # 2. Generate Heavy Mock Data (1000 bars)
    dates = pd.date_range("2024-01-01", periods=1000, freq="1min")
    market_data = pd.DataFrame(index=dates)
    
    # Create MultiIndex Columns like a real feed
    dfs = {}
    for tk in config['tickers']:
        df = pd.DataFrame(index=dates)
        df["Close"] = np.random.normal(100, 1, 1000).cumsum()
        df["High"] = df["Close"] + 0.1
        df["Low"] = df["Close"] - 0.1
        df["Volume"] = np.random.randint(1000, 50000, 1000)
        dfs[tk] = df
        
    market_data = pd.concat(dfs.values(), axis=1, keys=dfs.keys())
    print(f"✅ Generated Mock Data: {market_data.shape} (1000 bars x 5 tickers)")
    
    # 3. Warmup
    print("🔥 Warming up JIT/Caches...")
    for _ in range(5):
        strategy.generate_signals(market_data.tail(50))
        
    # 4. Benchmark Loop
    iterations = 100
    times = []
    
    print(f"⏱️  Running {iterations} iterations...")
    
    for _ in range(iterations):
        # Simulate receiving a new bar update (taking last 100 rows window)
        window = market_data.tail(100)
        
        t0 = time.perf_counter()
        _ = strategy.generate_signals(window)
        t1 = time.perf_counter()
        
        times.append((t1 - t0) * 1000.0) # ms
        
    # 5. Results
    avg_time = np.mean(times)
    p99_time = np.percentile(times, 99)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print("\n" + "="*40)
    print(f"📊 BENCHMARK RESULTS (5 Tickers)")
    print("="*40)
    print(f"Avg Latency:  {avg_time:.4f} ms")
    print(f"Min Latency:  {min_time:.4f} ms")
    print(f"Max Latency:  {max_time:.4f} ms")
    print(f"P99 Latency:  {p99_time:.4f} ms")
    print("="*40)
    
    if avg_time < 50:
        print("✅ SUCCESS: Latency is < 50ms (Institutional Grade)")
    else:
        print("⚠️  WARNING: Latency exceeded 50ms budget.")

if __name__ == "__main__":
    run_benchmark()
