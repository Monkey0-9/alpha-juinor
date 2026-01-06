import sys
import os
import logging
import time
import pandas as pd
import numpy as np
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.profiler import LatencyProfiler, SectionTimer, profile_ns
from data.collectors.data_router import DataRouter
from strategies.factory import StrategyFactory
from risk.engine import RiskManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Benchmark")

async def run_async_bench(iterations=5):
    profiler = LatencyProfiler()
    router = DataRouter()
    
    # Mock tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "SPY", "QQQ"]
    
    strategy = StrategyFactory.create_strategy({
        "type": "institutional",
        "tickers": tickers,
        "use_ml": False 
    })
    
    logger.info(f"Starting Async Performance Benchmark ({iterations} iterations)...")
    
    for i in range(iterations):
        logger.info(f"Iteration {i+1}/{iterations}")
        
        with SectionTimer("e2e_cycle_async"):
            # 1. Parallel Async Data Fetch
            with SectionTimer("data_fetch_all_async"):
                full_panel = await router.get_panel_async(tickers, "2025-12-01", "2025-12-05")
            
            if not full_panel.empty:
                # Parallel Async Price Fetch
                with SectionTimer("latest_prices_fetch_async"):
                    current_prices = await router.get_latest_prices_async(tickers)

                # 2. Signal Generation
                with SectionTimer("signal_generation"):
                    signals = strategy.generate_signals(full_panel)
            
            # 3. Decision & Risk simulation
            with SectionTimer("decision_and_risk"):
                await asyncio.sleep(0.01) 
                
        await asyncio.sleep(1) 
    
    profiler.report()
    
    # Save results to a file for comparison
    results = {}
    for component in profiler.metrics.keys():
        results[component] = profiler.get_stats(component)
    
    import json
    with open("output/async_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Async metrics saved to output/async_metrics.json")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    asyncio.run(run_async_bench(iterations=3))
