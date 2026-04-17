import time
import asyncio
import psutil
import os
import json
import numpy as np
from datetime import datetime, timedelta
from src.nexus.models.market import MarketBar
from src.nexus.models.trade import Order, OrderSide, OrderType, PortfolioState
from src.nexus.risk.rules import SectorConcentrationRule
from src.nexus.data.storage import MarketDataStore

async def benchmark_data_throughput():
    """Measures how many bars per second the storage layer can handle."""
    store = MarketDataStore(base_path="data/benchmark_cache")
    # Generate 50,000 mock bars
    bars = [
        MarketBar(
            symbol="TEST",
            timestamp=datetime.now() + timedelta(minutes=i),
            open=100.0, high=101.0, low=99.0, close=100.5, volume=1000
        ) for i in range(50000)
    ]
    
    start = time.perf_counter()
    store.save_bars(bars)
    duration = time.perf_counter() - start
    
    throughput = len(bars) / duration
    return {
        "metric": "Data Ingestion Throughput",
        "value": round(throughput, 2),
        "unit": "bars/sec",
        "duration_s": round(duration, 4)
    }

def benchmark_risk_latency():
    """Measures nanosecond-scale latency for institutional risk rules."""
    sector_map = {f"TICKER_{i}": "Tech" for i in range(1000)}
    rule = SectorConcentrationRule(sector_map=sector_map, max_sector_weight=0.20)
    portfolio = PortfolioState(cash=1000000, equity=1000000, positions={})
    order = Order(symbol="TICKER_1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100, limit_price=150.0)
    
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        rule.validate(order, portfolio)
        latencies.append(time.perf_counter_ns() - start)
    
    return {
        "metric": "Risk Validation Latency",
        "p50_ns": int(np.percentile(latencies, 50)),
        "p95_ns": int(np.percentile(latencies, 95)),
        "p99_ns": int(np.percentile(latencies, 99)),
        "unit": "ns"
    }

def benchmark_memory_footprint():
    """Measures baseline and peak memory usage of the engine process."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return {
        "metric": "Process Memory Usage",
        "value": round(mem_mb, 2),
        "unit": "MB"
    }

async def run_all_benchmarks():
    print("--- Starting Institutional Benchmarks ---")
    results = []
    
    results.append(await benchmark_data_throughput())
    results.append(benchmark_risk_latency())
    results.append(benchmark_memory_footprint())
    
    print(json.dumps(results, indent=2))
    
    # Save to artifact directory
    with open("benchmarks_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n[SUCCESS] Benchmarks complete. Report saved to benchmarks_report.json")

if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
