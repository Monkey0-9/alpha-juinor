"""
Verify Research Pipeline
========================
Tests the end-to-end flow of:
1. Hypothesis Manager (Registration)
2. Rapid Research Backtester (Execution)
3. Metrics Storage
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research.hypothesis_manager import HypothesisManager
from research.rapid_research_backtester import RapidResearchBacktester

def simple_momentum_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Hypothesis: 10-day Momentum.
    Signal = 1 if Return(10d) > 0 else -1
    """
    # Vectorized logic
    returns = data['Close'].pct_change(10)
    signals = returns.apply(lambda x: 1.0 if x > 0 else -1.0)
    return signals.fillna(0.0)

def main():
    # 1. Setup Data (Synthetic)
    print("Generating synthetic data...")
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    # Random walk with drift
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 500)
    price = 100 * (1 + returns).cumprod()

    market_data = pd.DataFrame({'Close': price}, index=dates)

    # 2. Register Hypothesis
    print("Registering hypothesis...")
    manager = HypothesisManager()
    hyp_id = manager.add_hypothesis(
        name="Synthetic Momentum",
        description="Verify pipeline with simple momentum",
        code="simple_momentum_strategy"
    )
    print(f"Hypothesis ID: {hyp_id}")

    # 3. Running Backtest
    print("Running backtest...")
    backtester = RapidResearchBacktester()
    metrics = backtester.run(simple_momentum_strategy, market_data)

    print("Backtest Metrics:", metrics)

    # 4. Update Database
    # Remove non-serializable objects (equity curve)
    if 'cumulative_returns' in metrics:
        del metrics['cumulative_returns']

    print("Updating metrics...")
    manager.update_metrics(hyp_id, metrics)

    # 5. Verify State
    hyp = manager.get_hypothesis(hyp_id)
    print(f"Final Status: {hyp.status}")
    print(f"Title: {hyp.name}")
    print(f"Sharpe: {hyp.metrics.get('sharpe')}")

if __name__ == "__main__":
    main()
