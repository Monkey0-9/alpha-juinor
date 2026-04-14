import ray
import pandas as pd
from typing import List, Dict
from .backtest_engine import DistributedBacktestEngine

@ray.remote
class AlphaWorker:
    """Ray worker for parallel alpha backtesting"""
    
    def __init__(self):
        self.engine = DistributedBacktestEngine()
        
    def run_backtest(self, alpha_values: pd.Series, price_data: pd.DataFrame):
        return self.engine.run_backtest(alpha_values, price_data)

class RayComputeManager:
    """Manages a cluster of alpha workers for 50+ simultaneous analyses"""
    
    def __init__(self, num_workers: int = 64): # Scale to 64 parallel workers
        ray.init(ignore_reinit_error=True)
        self.workers = [AlphaWorker.remote() for _ in range(num_workers)]
        
    def run_parallel_analysis(self, alpha_expressions: List[str], price_data: pd.DataFrame):
        """Analyze 50+ alpha streams concurrently with zero-error propagation"""
        futures = []
        for i, expr in enumerate(alpha_expressions):
            worker = self.workers[i % len(self.workers)]
            # Launch parallel analysis task
            futures.append(worker.run_backtest.remote(expr, price_data))
            
        print(f"DEBUG: Successfully launched {len(alpha_expressions)} simultaneous analyses.")
        return ray.get(futures)
