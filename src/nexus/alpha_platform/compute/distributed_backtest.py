import ray
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from mini_quant_fund.backtest.backtester import run_backtest

logger = logging.getLogger(__name__)

@ray.remote
def distributed_backtest_task(
    prices: pd.Series, 
    permissions: pd.Series, 
    capital: float = 1_000_000, 
    position_fraction: float = 0.25
) -> pd.DataFrame:
    """
    Individual backtest task executed on a Ray worker.
    """
    try:
        return run_backtest(prices, permissions, capital, position_fraction)
    except Exception as e:
        logger.error(f"Backtest task failed: {e}")
        return pd.DataFrame()

class DistributedBacktestRunner:
    """
    Ray-based distributed backtest runner for high-throughput alpha validation.
    """
    def __init__(self, address: str = None, num_cpus: int = None):
        if not ray.is_initialized():
            ray.init(address=address, num_cpus=num_cpus, ignore_reinit_error=True)
            logger.info("Ray initialized for distributed backtesting.")

    def run_parallel(self, backtest_configs: List[Dict[str, Any]]) -> List[pd.DataFrame]:
        """
        Executes multiple backtests in parallel across the Ray cluster.
        
        Args:
            backtest_configs: List of configurations for each backtest run.
                             Each config should contain 'prices' and 'permissions'.
        """
        futures = []
        for config in backtest_configs:
            prices = config.get('prices')
            permissions = config.get('permissions')
            capital = config.get('capital', 1_000_000)
            position_fraction = config.get('position_fraction', 0.25)
            
            if prices is None or permissions is None:
                logger.warning("Skipping config due to missing prices or permissions.")
                continue
                
            futures.append(
                distributed_backtest_task.remote(
                    prices, permissions, capital, position_fraction
                )
            )
            
        results = ray.get(futures)
        return results

    def shutdown(self):
        """Shuts down the Ray runtime."""
        ray.shutdown()
        logger.info("Ray runtime shut down.")

if __name__ == "__main__":
    # Example usage/test
    logging.basicConfig(level=logging.INFO)
    runner = DistributedBacktestRunner()
    
    # Generate some dummy data
    dates = pd.date_range("2020-01-01", periods=100)
    prices = pd.Series(np.cumsum(np.random.randn(100)) + 100, index=dates)
    
    configs = [
        {
            'prices': prices,
            'permissions': pd.Series(np.random.choice([0, 1], size=100), index=dates),
            'capital': 1_000_000
        } for _ in range(5)
    ]
    
    results = runner.run_parallel(configs)
    for i, res in enumerate(results):
        if not res.empty:
            print(f"Backtest {i} final equity: {res['equity'].iloc[-1]:.2f}")
    
    runner.shutdown()
