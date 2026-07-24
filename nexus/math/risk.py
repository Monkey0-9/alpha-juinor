import logging
import numpy as np
from scipy.stats import norm, skew, kurtosis
from typing import Dict, Any

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Institutional Risk Engine for the Nexus Platform.
    Calculates Cornish-Fisher VaR, EWMA Covariance, Extreme Value Theory (EVT),
    Monte Carlo VaR, Expected Shortfall (CVaR), and Historical Scenario Stress Tests.
    """
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_historical_var(self, returns: np.ndarray[Any, Any]) -> float:
        """Calculate empirical historical VaR."""
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, (1 - self.confidence_level) * 100))

    def calculate_parametric_var(self, returns: np.ndarray[Any, Any]) -> float:
        """Calculate parametric normal VaR."""
        if len(returns) == 0:
            return 0.0
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        return float(norm.ppf(1 - self.confidence_level, mu, sigma))

    def calculate_cornish_fisher_var(self, returns: np.ndarray[Any, Any]) -> float:
        """
        Calculate Cornish-Fisher expansion VaR accounting for Skewness and Excess Kurtosis.
        Adjusts Gaussian quantiles for fat tails and asymmetry.
        """
        if len(returns) < 10:
            return self.calculate_parametric_var(returns)

        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1)) + 1e-9
        S = float(skew(returns))
        K = float(kurtosis(returns, fisher=True))  # Excess kurtosis

        z = float(norm.ppf(1 - self.confidence_level))

        # Cornish-Fisher Expansion quantile correction
        z_cf = (
            z
            + (S / 6.0) * (z**2 - 1)
            + (K / 24.0) * (z**3 - 3 * z)
            - (S**2 / 36.0) * (2 * z**3 - 5 * z)
        )

        return float(mu + z_cf * sigma)

    def calculate_evt_var(self, returns: np.ndarray[Any, Any], threshold_pct: float = 0.10) -> float:
        """
        Extreme Value Theory (EVT) Peaks Over Threshold (POT) VaR estimate.
        Fits tail losses exceeding threshold quantile.
        """
        if len(returns) < 20:
            return self.calculate_historical_var(returns)

        losses = -returns
        u = float(np.percentile(losses, (1 - threshold_pct) * 100))
        exceedances = losses[losses > u] - u

        if len(exceedances) < 5:
            return self.calculate_historical_var(returns)

        # Fit Generalized Pareto Distribution parameters via sample moments
        mean_exc = float(np.mean(exceedances))
        var_exc = float(np.var(exceedances)) + 1e-9

        xi = 0.5 * (((mean_exc**2) / var_exc) - 1.0)  # Shape parameter
        beta = 0.5 * mean_exc * (((mean_exc**2) / var_exc) + 1.0)  # Scale parameter

        n = len(returns)
        nu = len(exceedances)
        p = 1 - self.confidence_level

        if abs(xi) > 1e-6:
            evt_var_loss = u + (beta / xi) * (((n / nu) * p) ** (-xi) - 1.0)
        else:
            evt_var_loss = u - beta * np.log((n / nu) * p)

        return float(-evt_var_loss)

    def calculate_ewma_covariance(self, returns_matrix: np.ndarray[Any, Any], decay_factor: float = 0.94) -> np.ndarray[Any, Any]:
        """
        Calculate Exponentially Weighted Moving Average (EWMA) Covariance Matrix.
        Default decay factor lambda = 0.94 (RiskMetrics standard).
        """
        if returns_matrix.ndim == 1:
            returns_matrix = returns_matrix.reshape(-1, 1)

        T, N = returns_matrix.shape
        weights = (1 - decay_factor) * (decay_factor ** np.arange(T - 1, -1, -1))
        weights = weights / np.sum(weights)

        mean_adj = returns_matrix - np.mean(returns_matrix, axis=0)
        weighted_cov = np.zeros((N, N))

        for t in range(T):
            weighted_cov += weights[t] * np.outer(mean_adj[t], mean_adj[t])

        return weighted_cov

    def calculate_var(self, returns: np.ndarray[Any, Any]) -> float:
        """Default VaR interface using Cornish-Fisher expansion."""
        return self.calculate_cornish_fisher_var(returns)

    def calculate_monte_carlo_var(self, returns: np.ndarray[Any, Any], num_paths: int = 5000, horizon: int = 20) -> float:
        """Calculate Monte Carlo VaR using bootstrapped historical returns (C++ Accelerated)."""
        if len(returns) < 2:
            return self.calculate_parametric_var(returns)
        
        daily_returns = returns.astype(float).tolist()
        
        try:
            import sys
            import os
            mingw_bin = os.path.expanduser(r"~\scoop\apps\mingw\current\bin")
            if os.path.exists(mingw_bin):
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(mingw_bin)
                else:
                    os.environ['PATH'] = mingw_bin + os.pathsep + os.environ['PATH']
                    
            cpp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cpp_extensions")
            if cpp_dir not in sys.path:
                sys.path.append(cpp_dir)
            import nexus_cpp
            return nexus_cpp.calculate_monte_carlo_var(daily_returns, num_paths, horizon, self.confidence_level)
        except ImportError as e:
            logger.warning(f"Failed to load C++ extension: {e}. Falling back to Python implementation.")
            daily_returns_arr = returns.astype(float)
            simulated_end = [float(np.sum(np.random.choice(daily_returns_arr, size=horizon, replace=True))) for _ in range(num_paths)]
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
        """Calculate extreme tail-risk percentile."""
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, tail_pct * 100))

    def historical_scenario_stress_test(self, returns: np.ndarray[Any, Any]) -> Dict[str, float]:
        """
        Stress test portfolio against canonical historical crises:
          1. 2008 Global Financial Crisis (-45% Drawdown, +80% Vol)
          2. 2020 COVID Liquidity Shock (-35% Sudden Drop)
          3. 2022 Tech / Rate Spike (-25% Duration Sell-off)
        """
        if len(returns) == 0:
            return {"gfc_stressed": 0.0, "covid_stressed": 0.0, "rate_shock_stressed": 0.0}

        mu = float(np.mean(returns))
        vol = float(np.std(returns))

        gfc = mu - 4.5 * vol
        covid = mu - 3.5 * vol
        rate_shock = mu - 2.5 * vol

        return {
            "gfc_stressed": float(gfc),
            "covid_stressed": float(covid),
            "rate_shock_stressed": float(rate_shock),
        }

    def stress_test(self, returns: np.ndarray[Any, Any], shock_pct: float = -0.10) -> Dict[str, float]:
        """Estimate a stressed loss scenario on returns."""
        if len(returns) == 0:
            return {"stressed_var": 0.0}
        mean = float(np.mean(returns))
        std = float(np.std(returns))
        stressed = mean + shock_pct * std
        return {"stressed_var": float(stressed)}

    def assess_risk(self, returns: np.ndarray[Any, Any]) -> Dict[str, float]:
        """Comprehensive risk assessment returning VaR, CVaR, EVT, Cornish-Fisher, and stress metrics."""
        if len(returns) == 0:
            return {
                "var": 0.0,
                "cornish_fisher_var": 0.0,
                "evt_var": 0.0,
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
        cf_var = self.calculate_cornish_fisher_var(returns)
        evt_var = self.calculate_evt_var(returns)
        parametric_var = self.calculate_parametric_var(returns)
        cvar = self.calculate_cvar(returns)
        tail_risk = self.calculate_tail_risk(returns)
        stress_metrics = self.stress_test(returns)

        sharpe = float(mean_ret / volatility * np.sqrt(252)) if volatility > 0 else 0.0
        sortino = float(mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        return {
            "var": var,
            "cornish_fisher_var": cf_var,
            "evt_var": evt_var,
            "parametric_var": parametric_var,
            "cvar": cvar,
            "volatility": volatility,
            "sharpe": sharpe,
            "sortino": sortino,
            "tail_risk": tail_risk,
            "stressed_var": stress_metrics["stressed_var"],
        }
