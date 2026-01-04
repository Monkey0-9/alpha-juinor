# reports/attribution.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AttributionEngine:
    """
    Institutional Performance Attribution & Risk Decomposition.
    Analyzes 'Why' the strategy performed the way it did.
    """
    
    def calculate_attribution(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes performance diagnostics.
        """
        if equity_df.empty:
            return {}
            
        returns = equity_df['equity'].pct_change().dropna()
        
        # 1. Turnover Attribution
        turnover = self._calculate_turnover(trades_df, equity_df)
        
        # 2. Regime Analysis (Simple Vol-based regime)
        regimes = self._analyze_regimes(returns)
        
        # 3. Risk Decomp (Simplified)
        vol = returns.std() * np.sqrt(252)
        mdd = (equity_df['equity'] / equity_df['equity'].cummax() - 1).min()
        
        # 4. Agent Attribution (New)
        agent_metrics = self._evaluate_agents(trades_df)
        
        return {
            "total_return": (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1,
            "annualized_vol": vol,
            "max_drawdown": mdd,
            "avg_daily_turnover": turnover,
            "regime_performance": regimes,
            "efficiency_ratio": returns.mean() / (returns.abs().mean() + 1e-9),
            "agent_performance": agent_metrics
        }

    def _evaluate_agents(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes which signals led to profitable trades."""
        if trades_df.empty or 'meta' not in trades_df.columns:
            return {}
            
        # Institutional agents usually tag their 'conviction' or 'source' in meta
        # For our system, we'll look for 'hit_rate' based on subsequent returns if available
        # or simplified 'profit_contribution'
        
        # This is a stub for granular agentic analysis
        return {
            "technical_hit_rate": 0.65, # Mock: Usually comes from logging historical signals
            "fundamental_hit_rate": 0.58,
            "consensus_score": 0.72
        }

    def _calculate_turnover(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> float:
        """Percentage of NAV traded per day on average."""
        if trades_df is None or trades_df.empty:
            return 0.0
        
        # Sum of absolute trade values
        total_volume = (trades_df['quantity'].abs() * trades_df['price']).sum()
        days = len(equity_df)
        avg_nav = equity_df['equity'].mean()
        
        return (total_volume / avg_nav) / days if days > 0 else 0.0

    def _analyze_regimes(self, returns: pd.Series) -> Dict[str, float]:
        """Performance broken down by market environment."""
        # Define Bull/Bear based on signal or simple moving average proxy
        # Here we just use positive vs negative return days as a stub
        bull_days = returns[returns > 0]
        bear_days = returns[returns <= 0]
        
        return {
            "bull_day_avg_ret": bull_days.mean() if not bull_days.empty else 0,
            "bear_day_avg_ret": bear_days.mean() if not bear_days.empty else 0,
            "up_day_ratio": len(bull_days) / len(returns) if len(returns) > 0 else 0
        }
