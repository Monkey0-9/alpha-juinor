
import time
import asyncio
import logging
import multiprocessing
from mini_quant_fund.core.engines.signal_engine import get_signal_engine
from mini_quant_fund.brokers.mock_broker import MockBroker
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("STRESS_TEST")

def run_load_worker(worker_id: int, iterations: int):
    """Heavy-duty worker for parallel load testing."""
    engine = get_signal_engine(["AAPL", "NVDA", "TSLA", "MSFT", "GOOG"])
    broker = MockBroker()
    
    # Mock MultiIndex data
    iterables = [["AAPL", "NVDA", "TSLA", "MSFT", "GOOG"], ["Open", "High", "Low", "Close", "Volume"]]
    index = pd.MultiIndex.from_product(iterables)
    mock_data = pd.DataFrame(np.random.randn(10, 25), columns=index)

    start = time.time()
    count = 0
    for _ in range(iterations):
        # Full Cycle: Signal -> Brain -> Execution
        signals = engine.generate_signals(mock_data)
        for sym in signals:
            broker.submit_order(symbol=sym, qty=100, side="buy")
        count += 1
    
    end = time.time()
    rps = count / (end - start)
    return rps

def main():
    num_workers = multiprocessing.cpu_count()
    iterations_per_worker = 100
    
    logger.info(f"Starting Institutional Stress Test with {num_workers} workers...")
    
    pool = multiprocessing.Pool(processes=num_workers)
    results = pool.starmap(run_load_worker, [(i, iterations_per_worker) for i in range(num_workers)])
    
    total_rps = sum(results)
    avg_rps_per_worker = total_rps / num_workers
    
    logger.info("=" * 50)
    logger.info("INSTITUTIONAL LOAD TEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total Workers (Cores): {num_workers}")
    logger.info(f"Total Aggregated RPS: {total_rps:.2f}")
    logger.info(f"Avg RPS per Worker:    {avg_rps_per_worker:.2f}")
    logger.info(f"Pass/Fail (1000 RPS):  {'PASS' if total_rps >= 1000 else 'FAIL'}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
