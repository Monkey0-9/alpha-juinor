
import time
import logging
import numpy as np
from typing import Dict, List
from core.engines.signal_engine import get_signal_engine
from brokers.mock_broker import MockBroker
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LATENCY_PROFILER")

class LatencyProfiler:
    """
    Measures 'Tick-to-Trade' latency with nanosecond precision.
    Bridges the gap from 'Theoretical Claims' to 'Validated Performance'.
    """
    def __init__(self, iterations: int = 100):
        self.iterations = iterations
        self.engine = get_signal_engine(["AAPL"])
        self.broker = MockBroker()
        
        # Prepare mock data once to isolate execution latency
        iterables = [["AAPL"], ["Open", "High", "Low", "Close", "Volume"]]
        index = pd.MultiIndex.from_product(iterables)
        self.mock_market_data = pd.DataFrame(np.random.randn(5, 5), columns=index)

    def run_profile(self):
        logger.info(f"Starting End-to-End Latency Profiling ({self.iterations} iterations)...")
        latencies_ns = []

        for _ in range(self.iterations):
            start_ns = time.perf_counter_ns()
            
            # 1. Signal Generation
            signals = self.engine.generate_signals(self.mock_market_data)
            
            # 2. Execution Logic
            if "AAPL" in signals:
                self.broker.submit_order(symbol="AAPL", qty=10, side="buy")
            
            end_ns = time.perf_counter_ns()
            latencies_ns.append(end_ns - start_ns)

        latencies_ms = np.array(latencies_ns) / 1e6
        
        logger.info("=" * 40)
        logger.info("TICK-TO-TRADE LATENCY RESULTS")
        logger.info("=" * 40)
        logger.info(f"Mean Latency:   {np.mean(latencies_ms):.4f} ms")
        logger.info(f"Median Latency: {np.median(latencies_ms):.4f} ms")
        logger.info(f"P95 Latency:    {np.percentile(latencies_ms, 95):.4f} ms")
        logger.info(f"P99 Latency:    {np.percentile(latencies_ms, 99):.4f} ms")
        logger.info(f"Theoretical RPS: {1000 / np.mean(latencies_ms):.0f}")
        logger.info("=" * 40)

if __name__ == "__main__":
    profiler = LatencyProfiler()
    profiler.run_profile()
