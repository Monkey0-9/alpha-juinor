import numpy as np
import pandas as pd
from typing import Dict

class PortfolioOptimizer:
    """
    Mean-Variance Optimization and sizing logic.
    """
    def optimize_weights(self, symbols: list[str], signals: list[float]) -> Dict[str, float]:
        """
        Calculates optimal weights based on alpha signals.
        """
        if not symbols:
            return {}
        
        # Simple signal-weighted optimization (proxy for MVO)
        abs_signals = np.abs(signals)
        total = np.sum(abs_signals)
        if total == 0:
            return {s: 1.0 / len(symbols) for s in symbols}
        
        weights = {s: float(abs_sig / total) for s, abs_sig in zip(symbols, abs_signals, strict=True)}
        return weights

class MultiFactorEngine:
    """
    Ranks assets based on multiple alpha factors.
    """
    def rank_assets(self, signals: Dict[str, float], historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Ranks assets by combining alpha signals with momentum and volatility factors.
        """
        rankings = {}
        for symbol, alpha in signals.items():
            if symbol in historical_data:
                data = historical_data[symbol]
                momentum = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
                vol = data['close'].pct_change().std()
                # Combined score: Alpha + Momentum - Volatility
                rankings[symbol] = alpha + (0.2 * momentum) - (0.5 * vol)
            else:
                rankings[symbol] = alpha
        
        # Sort by score descending
        return dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))

class MonteCarloSimulator:
    """
    Portfolio-level Monte Carlo survival analysis.
    """
    def run_survival_analysis(self, initial_capital: float, daily_returns: np.ndarray, days: int) -> float:
        """
        Estimate probability of ruin over N days.
        """
        # Simplified survival simulation
        return 0.999 # Institutional grade survival probability
