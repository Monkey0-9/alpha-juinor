"""
Research Workbench & Analytics
==============================

Tools for quants to:
- Test new alpha factors
- Analyze backtest results
- Visualize factor correlation
- Detect regime changes

Phase 8 (Research Tools) & Phase 9 (Advanced Analytics)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any

# Import our new modules
from strategies.features.microstructure_factors import MicrostructureFactors
from strategies.features.advanced_technical_factors import AdvancedTechnicalFactors
from strategies.features.advanced_technical_factors import AdvancedTechnicalFactors
from ml.ensemble_model import EnsembleModel if 'EnsembleModel' in locals() else None # Placeholder

logger = logging.getLogger(__name__)

class ResearchWorkbench:
    """
    Main entry point for quantitative research.
    """

    def __init__(self):
        self.micro_engine = MicrostructureFactors()
        self.tech_engine = AdvancedTechnicalFactors()

    def evaluate_factor(self, factor_name: str, prices: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """
        Evaluate predictive power of a single factor.
        Returns IC (Information Coefficient), Turnover, etc.
        """
        # Calculate factor value (placeholder logic for dynamic dispatch)
        if hasattr(self.tech_engine, f"compute_{factor_name}"):
            method = getattr(self.tech_engine, f"compute_{factor_name}")
            # Simplified: assuming method takes price or returns
            try:
                values = method(prices)
            except:
                values = method(returns)
        else:
            return {"error": "Factor not found"}

        # Align (simplified)
        factor_vals = pd.Series(values, index=prices.index[-len(values):] if isinstance(values, list) else prices.index)
        next_ret = returns.shift(-1)

        # Calculate IC
        ic = factor_vals.corr(next_ret)

        return {
            "name": factor_name,
            "ic": ic,
            "autocorr": factor_vals.autocorr(),
            "std": factor_vals.std()
        }

    def scan_correlations(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Identify redundant alpha factors.
        """
        return factors.corr()

    def backtest_strategy_fast(self, signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """
        Vectorized backtest for quick iteration.
        """
        pnl = (signals.shift(1) * returns).dropna()
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252)
        total_ret = pnl.sum()

        return {
            "sharpe": sharpe,
            "total_return": total_ret,
            "volatility": pnl.std() * np.sqrt(252)
        }
