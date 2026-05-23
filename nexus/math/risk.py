import logging
import numpy as np
from scipy.stats import norm
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Institutional Risk Engine for the Nexus Platform.
    Calculates historical VaR, Monte Carlo VaR, CVaR, and stress metrics.
    """
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_historical_var(self, returns: np.ndarray[Any, Any]) -> float:
        """Calculate empirical historical VaR."""
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, (1 - self.confidence_level) * 100))

    def calculate_parametric_var(self, returns: np.ndarray[Any, Any]) -> float:
        """Calculate parametric normal VaR as a fallback."""
        if len(returns) == 0:
            return 0.0
        mu = np.mean(returns)
        sigma = np.std(returns)
        return float(norm.ppf(1 - self.confidence_level, mu, sigma))

    def calculate_var(self, returns: np.ndarray[Any, Any]) -> float:
        """Legacy VaR interface compatible with existing tests and code."""
        return self.calculate_historical_var(returns)

    def calculate_monte_carlo_var(self, returns: np.ndarray[Any, Any], num_paths: int = 5000, horizon: int = 20) -> float:
        """Calculate Monte Carlo VaR using bootstrapped historical returns."""
        if len(returns) < 2:
            return self.calculate_parametric_var(returns)
        daily_returns = returns.astype(float)
        simulated_end = []
        for _ in range(num_paths):
            path = np.random.choice(daily_returns, size=horizon, replace=True)
            simulated_end.append(np.sum(path))
        return float(np.percentile(simulated_end, (1 - self.confidence_level) * 100))

    def calculate_cvar(self, returns: np.ndarray[Any, Any]) -> float:
        """Calculate Conditional VaR / Expected Shortfall."""
        if len(returns) == 0:
            return 0.0
        var = self.calculate_historical_var(returns)
        tail_losses = returns[returns <= var]
        if len(tail_losses) == 0:
            return var
        return float(np.mean(tail_losses))

    def calculate_tail_risk(self, returns: np.ndarray[Any, Any], tail_pct: float = 0.01) -> float:
        """Calculate a more extreme tail-risk percentile."""
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, tail_pct * 100))

    def stress_test(self, returns: np.ndarray[Any, Any], shock_pct: float = -0.10) -> Dict[str, float]:
        """Estimate a stressed loss scenario on returns."""
        if len(returns) == 0:
            return {"stressed_var": 0.0}
        mean = np.mean(returns)
        std = np.std(returns)
        stressed = mean + shock_pct * std
        return {"stressed_var": float(stressed)}

    def assess_risk(self, returns: np.ndarray[Any, Any]) -> Dict[str, float]:
        """Comprehensive risk assessment returning VaR, CVaR, and other metrics."""
        if len(returns) == 0:
            return {
                "var": 0.0,
                "parametric_var": 0.0,
                "cvar": 0.0,
                "volatility": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "tail_risk": 0.0,
                "stressed_var": 0.0,
            }

        volatility = float(np.std(returns, ddof=1))
        downside = returns[returns < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 0 else 0.0
        mean_ret = float(np.mean(returns))

        var = self.calculate_historical_var(returns)
        parametric_var = self.calculate_parametric_var(returns)
        cvar = self.calculate_cvar(returns)
        tail_risk = self.calculate_tail_risk(returns)
        stress_metrics = self.stress_test(returns)

        sharpe = float(mean_ret / volatility * np.sqrt(252)) if volatility > 0 else 0.0
        sortino = float(mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        return {
            "var": var,
            "parametric_var": parametric_var,
            "cvar": cvar,
            "volatility": volatility,
            "sharpe": sharpe,
            "sortino": sortino,
            "tail_risk": tail_risk,
            "stressed_var": stress_metrics["stressed_var"],
        }
