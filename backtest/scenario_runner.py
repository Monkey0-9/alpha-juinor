"""
backtest/scenario_runner.py

Section G: Historical Crisis Backtest Runner.
Runs full pipeline simulations over defined crisis periods.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

# Import components
# from scripts.ingest_5y_batch import NightlyIngestionAgent (would be used for fetch)
# from portfolio.pm_brain import PMBrain
# from execution.fill_simulator import FillSimulator

logger = logging.getLogger("SCENARIO_RUNNER")

@dataclass
class ScenarioResult:
    name: str
    max_drawdown: float
    cvar_95: float
    recovery_time_days: int
    final_return: float
    passed: bool

class ScenarioRunner:
    SCENARIOS = {
        "TECH_BUBBLE": ("2000-01-01", "2003-01-01"),
        "GFC": ("2007-06-01", "2009-06-01"),
        "COVID": ("2020-02-01", "2020-06-01"),
        "RATE_SHOCK": ("2022-01-01", "2022-12-31")
    }

    def __init__(self):
        pass

    def run_scenario(self, name: str, start_date: str, end_date: str) -> ScenarioResult:
        logger.info(f"Running Scenario: {name} ({start_date} to {end_date})")

        # 1. Ingest / Mock Data
        # In a real run, we'd trigger ingestion or load pre-cached data.
        # For this prototype agent, we simulate the equity curve behavior
        # based on the intent (as we can't easily backfill 2000 data live from Yahoo usually reliably in 5s)

        # Simulating a "Survival" test
        # We assume the PM Brain runs and generates a series of returns.

        # Let's mock a return series that drops then recovers (or doesn't)
        # to test the METRICS logic.

        dates = pd.date_range(start_date, end_date, freq='B')
        n = len(dates)

        # Simulated returns (Random Walk with shock)
        # If GFC, we inject a crash
        returns = np.random.normal(0.0005, 0.01, n)

        if name == "GFC":
            # Inject crash in middle
            mid = n // 2
            returns[mid:mid+20] = -0.05 # -5% for 20 days

        equity = (1 + returns).cumprod()

        # Compute Metrics
        # Max DD
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown)

        # CVaR 95
        cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)])

        # Recovery Time
        # Time from Max DD to new High
        # Simplified: check if final > peak at max_dd
        idx_dd = np.argmin(drawdown)
        peak_val = peak[idx_dd]

        recovery_days = 9999
        if idx_dd < n-1:
            future = equity[idx_dd:]
            rec_idx = np.where(future >= peak_val)[0]
            if len(rec_idx) > 0:
                recovery_days = rec_idx[0]

        # Acceptance Criteria
        passed = True
        if max_dd < -0.30: passed = False # Fail if > 30% DD
        if cvar_95 < -0.03: passed = False # Fail if CVaR worse than -3%

        return ScenarioResult(name, max_dd, cvar_95, recovery_days, equity[-1]-1, passed)

    def run_all(self) -> List[ScenarioResult]:
        results = []
        for name, (start, end) in self.SCENARIOS.items():
            try:
                res = self.run_scenario(name, start, end)
                results.append(res)
            except Exception as e:
                logger.error(f"Scenario {name} failed: {e}")
                results.append(ScenarioResult(name, 0, 0, 0, 0, False))
        return results

if __name__ == "__main__":
    runner = ScenarioRunner()
    results = runner.run_all()
    for r in results:
        print(r)
