"""
Rapid Research Backtester
=========================

A high-performance, vectorized backtesting engine designed for research.
Prioritizes speed and statistical validation over granular execution simulation.

Features:
- Vectorized Signal Generation & PnL Calculation.
- Standardized Performance Metrics (Sharpe, Sortino, MaxDD).
- "Friction" simulation (transaction costs).
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class RapidResearchBacktester:
    def __init__(self, transaction_cost_bps: float = 5.0):
        self.transaction_cost_bps = transaction_cost_bps

    def run(self,
            signal_func: Callable[[pd.DataFrame], pd.DataFrame],
            market_data: pd.DataFrame,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> Dict:
        """
        Run a vectorized backtest for a single asset or universe.

        Args:
            signal_func: Function taking market_data -> signal DataFrame (weights or -1/0/1).
                         Signal at index `t` is assumed to act on Return at `t+1` (or `t` if logic handles lag).
                         We enforce shift(1) by default to prevent lookahead if configured.
            market_data: DataFrame with 'Close' (and 'Open'/'Volume' if needed).

        Returns:
            Dictionary of performance metrics.
        """
        # 1. Data Prep
        data = market_data.copy()
        if start_date:
            data = data.loc[start_date:]
        if end_date:
            data = data.loc[:end_date]

        if data.empty:
            return {"error": "No data for backtest"}

        # 2. Generate Signals (Vectorized)
        try:
            # User provided function to generate signals
            # Signal should be target position/weight
            signals = signal_func(data)
        except Exception as e:
            logger.error(f"Signal Generation Failed: {e}")
            return {"error": str(e)}

        if signals.empty:
             return {"error": "Empty signals generated"}

        # 3. Calculate Returns
        # Asset Returns
        if 'Close' in data.columns:
            element_rets = data['Close'].pct_change()
        else:
            # Handle MultiIndex if necessary, or error
            # For MVP assume single asset or aligned DataFrame
            return {"error": "Data must contain 'Close' column"}

        # Align lengths
        # Strategy Return = Signal(t-1) * Return(t)
        # We shift signal by 1 to simulate "Decision at Close t, Realize Return at Close t+1"
        aligned_signals = signals.shift(1)

        # Strategy Returns (Raw)
        strategy_rets = aligned_signals * element_rets

        # 4. Apply Transaction Costs
        # Cost = |Signal(t) - Signal(t-1)| * Cost_Bps
        trades = signals.diff().abs().fillna(0.0)
        costs = trades * (self.transaction_cost_bps / 10000.0)

        # Net Returns
        net_strategy_rets = strategy_rets - costs.shift(1) # Cost incurred at trade time, affects PnL
        # Correction: Cost reduces equity.
        # R_net = R_gross - Cost
        # Since we are doing bps of notional, this approx works.
        net_strategy_rets = net_strategy_rets.fillna(0.0)

        # 5. Calculate Metrics
        metrics = self._calculate_metrics(net_strategy_rets)

        # Attach equity curve for plotting
        metrics['cumulative_returns'] = (1 + net_strategy_rets).cumprod()

        return metrics

    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate Sharpe, Sortino, Drawdown, etc."""
        if returns.empty or returns.sum() == 0:
             return {
                 "sharpe": 0.0,
                 "annualized_return": 0.0,
                 "max_drawdown": 0.0,
                 "win_rate": 0.0
             }

        # Annualization Factor
        ann_factor = 252

        # 1. Annualized Return
        # Compounded
        cum_ret = (1 + returns).cumprod().iloc[-1]
        n_years = len(returns) / ann_factor
        if n_years < 0.1: n_years = 0.1 # Avoid div by zero small sample
        cagr = (cum_ret ** (1 / n_years)) - 1

        # 2. Sharpe Ratio
        mu = returns.mean() * ann_factor
        sigma = returns.std() * np.sqrt(ann_factor)
        sharpe = mu / sigma if sigma > 0 else 0.0

        # 3. Sortino Ratio (Downside Deviation)
        downside = returns[returns < 0]
        sigma_down = downside.std() * np.sqrt(ann_factor)
        sortino = mu / sigma_down if sigma_down > 0 else 0.0

        # 4. Max Drawdown
        cum_equity = (1 + returns).cumprod()
        running_max = cum_equity.cummax()
        drawdown = (cum_equity - running_max) / running_max
        max_dd = drawdown.min()

        return {
            "sharpe": float(sharpe) if not np.isnan(sharpe) else 0.0,
            "sortino": float(sortino) if not np.isnan(sortino) else 0.0,
            "cagr": float(cagr) if not np.isnan(cagr) else 0.0,
            "max_drawdown": float(max_dd) if not np.isnan(max_dd) else 0.0,
            "volatility": float(sigma) if not np.isnan(sigma) else 0.0
        }
