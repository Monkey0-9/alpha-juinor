
"""
research/monte_carlo.py

OFFLINE RESEARCH MODULE.
NEVER IMPORT INTO LIVE ENGINE.

Purpose:
Run Monte Carlo simulations to stress test strategies against
shuffled/synthetic market data.
"""

import numpy as np
import pandas as pd

def run_monte_carlo(returns: pd.Series, n_sims: int = 1000, n_days: int = 252):
    """
    Bootstrap Monte Carlo Simulation.
    """
    if returns.empty:
        return {}
    
    simulations = []
    
    for i in range(n_sims):
        # Sample with replacement
        sim_rets = np.random.choice(returns, size=n_days, replace=True)
        cum_rets = np.nancumprod(1 + sim_rets)
        simulations.append(cum_rets)
    
    simulations = np.array(simulations)
    
    # Calculate stats
    final_values = simulations[:, -1]
    
    results = {
        "mean_final": np.mean(final_values),
        "median_final": np.median(final_values),
        "worst_case_1pct": np.percentile(final_values, 1),
        "best_case_99pct": np.percentile(final_values, 99)
    }
    
    return results

if __name__ == "__main__":
    # Example usage for research
    print("Monte Carlo Module Loaded. Run for research only.")
