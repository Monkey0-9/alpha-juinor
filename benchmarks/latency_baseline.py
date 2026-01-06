import sys
import os
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.profiler import LatencyProfiler, SectionTimer, profile_ns
from data.collectors.data_router import DataRouter
from strategies.factory import StrategyFactory
from risk.engine import RiskManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Benchmark")

def run_baseline_bench(iterations=5):
    profiler = LatencyProfiler()
    router = DataRouter()
    
    # Mock tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "SPY", "QQQ"]
    
    strategy = StrategyFactory.create_strategy({
        "type": "institutional",
        "tickers": tickers,
        "use_ml": False # Baseline without ML first to isolate infra
    })
    
    logger.info(f"Starting Baseline Benchmark ({iterations} iterations)...")
    
    for i in range(iterations):
        logger.info(f"Iteration {i+1}/{iterations}")
        
        with SectionTimer("e2e_cycle"):
            # 1. Data Fetch
            with SectionTimer("data_fetch_all"):
                # Simulate the smart router fetch
                data_panel = {}
                for tk in tickers:
                    with SectionTimer(f"data_fetch_{tk}"):
                         # We use a small window for benchmark speed
                         df = router.get_price_history(tk, "2025-12-01", "2025-12-05")
                         data_panel[tk] = df
            
            # Convert to multi-index panel format expected by strategy
            with SectionTimer("panel_construction"):
                dfs = []
                for tk, df in data_panel.items():
                    if not df.empty:
                        df.columns = pd.MultiIndex.from_product([[tk], df.columns])
                        dfs.append(df)
                full_panel = pd.concat(dfs, axis=1)
                current_prices = {tk: full_panel[tk]['Close'].iloc[-1] for tk in data_panel if not data_panel[tk].empty}

            # 2. Signal Generation
            with SectionTimer("signal_generation"):
                signals = strategy.generate_signals(full_panel)
            
            # 3. Decision & Risk
            with SectionTimer("decision_and_risk"):
                # Simulate the allocation/risk path
                # (Simplified for baseline as full setup requires portfolio state)
                time.sleep(0.01) # Baseline overhead simulation
                
        time.sleep(1) # Gap between iterations
    
    profiler.report()
    
    # Save results to a file for comparison
    results = {}
    for component in profiler.metrics.keys():
        results[component] = profiler.get_stats(component)
    
    import json
    with open("output/baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Baseline metrics saved to output/baseline_metrics.json")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    run_baseline_bench(iterations=3)
