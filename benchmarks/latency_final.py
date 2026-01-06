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
logger = logging.getLogger("FinalBenchmark")

async def run_final_bench(iterations=5):
    profiler = LatencyProfiler()
    router = DataRouter()
    
    # 10 tickers to test parallel scaling
    tickers = ["AAPL", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "SPY", "QQQ", "TSLA"]
    
    strategy = StrategyFactory.create_strategy({
        "type": "institutional",
        "tickers": tickers,
        "use_ml": False 
    })
    
    logger.info("Starting Final Performance Benchmark...")
    
    # 1. Warmup WebSockets (Phase 2 feature)
    logger.info("Warming up WebSocket streams (60s wait for ticks)...")
    await router.start_streaming_async(tickers)
    await asyncio.sleep(5) # Real ticks take time, Mock might not work here but we test the code path
    
    for i in range(iterations):
        logger.info(f"Iteration {i+1}/{iterations}")
        
        with SectionTimer("e2e_cycle_final"):
            # 1. Parallel Async History Fetch (Phase 1 + 3)
            with SectionTimer("data_panel_fetch_optimized"):
                full_panel = await router.get_panel_async(tickers, "2025-12-01", "2025-12-05")
            
            if not full_panel.empty:
                # 2. Reactive Price Fetch (Phase 2)
                with SectionTimer("latest_prices_fetch_reactive"):
                    current_prices = await router.get_latest_prices_async(tickers)

                # 3. Parallel Signal Generation (Phase 3)
                with SectionTimer("signal_generation_parallel"):
                    signals = strategy.generate_signals(full_panel)
            
            # 4. Decision & Risk simulation
            with SectionTimer("decision_and_risk"):
                await asyncio.sleep(0.01) 
                
        await asyncio.sleep(1) 
    
    profiler.report()
    
    # Save results
    results = {component: profiler.get_stats(component) for component in profiler.metrics.keys()}
    with open("output/final_metrics.json", "w") as f:
        import json
        json.dump(results, f, indent=4)
    logger.info("Final metrics saved to output/final_metrics.json")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    asyncio.run(run_final_bench(iterations=3))
