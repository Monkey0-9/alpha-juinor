"""
backtest/scenario_runner.py

Historical Scenario and Stress Test Runner.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger("SCENARIO_RUNNER")


@dataclass
class ScenarioResult:
    """Result of a scenario run."""
    scenario_name: str
    start_date: str
    end_date: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    trades_count: int
    win_rate: float
    avg_holding_period: float
    meta: Dict[str, Any]


class ScenarioRunner:
    """
    Run historical scenarios and stress tests.

    Built-in scenarios:
    - COVID_CRASH: Feb-Mar 2020
    - TECH_SELLOFF: Sep 2022
    - FLASH_CRASH: May 2010
    - STEADY_BULL: 2017
    """

    BUILT_IN_SCENARIOS = {
        "COVID_CRASH": ("2020-02-19", "2020-03-23"),
        "TECH_SELLOFF": ("2022-08-15", "2022-10-12"),
        "FLASH_CRASH": ("2010-05-05", "2010-05-07"),
        "STEADY_BULL": ("2017-01-01", "2017-12-31"),
        "RATE_HIKE_2022": ("2022-01-01", "2022-06-30"),
    }

    def __init__(self, strategy_fn: Optional[Callable] = None):
        """
        Args:
            strategy_fn: Function(prices_df) -> signals_df
        """
        self.strategy_fn = strategy_fn
        self._results: List[ScenarioResult] = []
        logger.info("[SCENARIO_RUNNER] Initialized")

    def run_scenario(
        self,
        scenario_name: str,
        prices: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> ScenarioResult:
        """Run a single scenario."""
        # Use built-in dates if scenario is known
        if scenario_name in self.BUILT_IN_SCENARIOS and not start_date:
            start_date, end_date = self.BUILT_IN_SCENARIOS[scenario_name]

        # Filter data
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]

        if prices.empty:
            logger.warning(f"[SCENARIO] {scenario_name}: No data")
            return self._empty_result(scenario_name)

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Simple strategy simulation (buy-and-hold if no strategy_fn)
        if self.strategy_fn:
            signals = self.strategy_fn(prices)
            portfolio_returns = (signals.shift(1) * returns).sum(axis=1)
        else:
            # Equal weight buy-and-hold
            portfolio_returns = returns.mean(axis=1)

        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = (
            (portfolio_returns.mean() * 252) / volatility
            if volatility > 0 else 0
        )

        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        result = ScenarioResult(
            scenario_name=scenario_name,
            start_date=str(prices.index[0]),
            end_date=str(prices.index[-1]),
            total_return=float(total_return),
            max_drawdown=float(max_dd),
            sharpe_ratio=float(sharpe),
            volatility=float(volatility),
            trades_count=len(prices),
            win_rate=float((portfolio_returns > 0).mean()),
            avg_holding_period=1.0,
            meta={"days": len(prices)}
        )

        self._results.append(result)
        logger.info(
            f"[SCENARIO] {scenario_name}: Return={total_return:.2%}, "
            f"MaxDD={max_dd:.2%}, Sharpe={sharpe:.2f}"
        )
        return result

    def run_all_built_in(self, prices: pd.DataFrame) -> List[ScenarioResult]:
        """Run all built-in scenarios."""
        results = []
        for name in self.BUILT_IN_SCENARIOS:
            try:
                result = self.run_scenario(name, prices)
                results.append(result)
            except Exception as e:
                logger.error(f"[SCENARIO] {name} failed: {e}")
        return results

    def stress_test(
        self,
        prices: pd.DataFrame,
        shock_pct: float = -0.10
    ) -> ScenarioResult:
        """Apply synthetic shock and measure impact."""
        # Simulate a sudden drop
        shocked = prices.copy()
        mid_idx = len(shocked) // 2
        shocked.iloc[mid_idx:] *= (1 + shock_pct)

        return self.run_scenario(
            f"STRESS_SHOCK_{int(abs(shock_pct)*100)}pct",
            shocked
        )

    def _empty_result(self, name: str) -> ScenarioResult:
        return ScenarioResult(
            scenario_name=name,
            start_date="",
            end_date="",
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            volatility=0.0,
            trades_count=0,
            win_rate=0.0,
            avg_holding_period=0.0,
            meta={"error": "No data"}
        )

    def get_summary(self) -> pd.DataFrame:
        """Get summary of all scenario results."""
        if not self._results:
            return pd.DataFrame()

        data = [
            {
                "Scenario": r.scenario_name,
                "Return": f"{r.total_return:.2%}",
                "MaxDD": f"{r.max_drawdown:.2%}",
                "Sharpe": f"{r.sharpe_ratio:.2f}",
                "WinRate": f"{r.win_rate:.2%}"
            }
            for r in self._results
        ]
        return pd.DataFrame(data)
