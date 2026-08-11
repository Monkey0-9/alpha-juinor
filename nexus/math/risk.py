import logging
import numpy as np
from scipy.stats import norm, skew, kurtosis
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RiskEngine:
    def __init__(self, confidence_level: float = 0.95, ewma_lambda: float = 0.94):
        self.confidence_level = confidence_level
        self.ewma_lambda = ewma_lambda

    def calculate_ewma_covariance(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        n = len(returns)
        weights = np.array([(1 - self.ewma_lambda) * (self.ewma_lambda ** (n - 1 - i)) for i in range(n)])
        weights /= weights.sum()
        mean = np.average(returns, weights=weights)
        variance = np.average((returns - mean) ** 2, weights=weights)
        return variance

    def calculate_historical_var(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, (1 - self.confidence_level) * 100))

    def calculate_var(self, returns: np.ndarray, method: str = 'historical') -> float:
        if method == 'cornish_fisher':
            return self.calculate_cornish_fisher_var(returns)
        elif method == 'parametric':
            return self.calculate_parametric_var(returns)
        elif method == 'ewma':
            return self.calculate_ewma_var(returns)
        return self.calculate_historical_var(returns)

    def calculate_parametric_var(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        return float(norm.ppf(1 - self.confidence_level, mu, sigma))

    def calculate_cornish_fisher_var(self, returns: np.ndarray) -> float:
        if len(returns) < 5:
            return self.calculate_parametric_var(returns)
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        sk = skew(returns)
        ku = kurtosis(returns, fisher=True)
        z = norm.ppf(1 - self.confidence_level)
        z_cf = z + (sk / 6) * (z ** 2 - 1) + (ku / 24) * (z ** 3 - 3 * z) - (sk ** 2 / 36) * (2 * z ** 3 - 5 * z)
        return float(mu + z_cf * sigma)

    def calculate_ewma_var(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return self.calculate_parametric_var(returns)
        ewma_var = self.calculate_ewma_covariance(returns)
        z = norm.ppf(1 - self.confidence_level)
        return float(z * np.sqrt(ewma_var))

    def calculate_monte_carlo_var(self, returns: np.ndarray, num_paths: int = 10000, horizon: int = 20) -> float:
        if len(returns) < 2:
            return self.calculate_parametric_var(returns)
        daily_returns = returns.astype(float)
        try:
            import sys, os
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
            return nexus_cpp.calculate_monte_carlo_var(daily_returns.tolist(), num_paths, horizon, self.confidence_level)
        except ImportError:
            simulated_end = np.array([np.sum(np.random.choice(daily_returns, size=horizon, replace=True)) for _ in range(num_paths)])
            return float(np.percentile(simulated_end, (1 - self.confidence_level) * 100))

    def calculate_cvar(self, returns: np.ndarray, method: str = 'historical') -> float:
        if len(returns) == 0:
            return 0.0
        if method == 'cornish_fisher':
            var = self.calculate_cornish_fisher_var(returns)
        elif method == 'ewma':
            var = self.calculate_ewma_var(returns)
        else:
            var = self.calculate_historical_var(returns)
        tail_losses = returns[returns <= var]
        if len(tail_losses) == 0:
            return var
        return float(np.mean(tail_losses))

    def calculate_expected_shortfall(self, returns: np.ndarray, confidence: Optional[float] = None) -> float:
        conf = confidence or self.confidence_level
        var = np.percentile(returns, (1 - conf) * 100)
        tail = returns[returns <= var]
        return float(np.mean(tail)) if len(tail) > 0 else var

    def calculate_tail_risk(self, returns: np.ndarray, tail_pct: float = 0.01) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, tail_pct * 100))

    def calculate_extreme_value_var(self, returns: np.ndarray, threshold: Optional[float] = None) -> float:
        if len(returns) < 20:
            return self.calculate_historical_var(returns)
        threshold = threshold or float(np.percentile(np.abs(returns), 90))
        extremes = returns[returns < -threshold]
        if len(extremes) < 3:
            return self.calculate_historical_var(returns)
        xi = len(extremes) / len(returns)
        tail_mean = np.mean(extremes)
        var_pot = tail_mean + (threshold / xi) * (((1 - self.confidence_level) / xi) ** (-xi) - 1)
        return float(var_pot)

    def stress_test(self, returns: np.ndarray, scenarios: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if len(returns) == 0:
            return {"base_var": 0.0}
        base_var = self.calculate_historical_var(returns)
        results = {"base_var": base_var, "base_cvar": self.calculate_cvar(returns)}
        default_scenarios = {
            "2008_crisis": -0.15, "covid_crash": -0.12, "flash_crash": -0.08,
            "rate_hike_shock": -0.05, "mild_correction": -0.03,
        }
        scenarios = scenarios or default_scenarios
        for name, shock in scenarios.items():
            stressed = returns + np.random.normal(shock, 0.02, len(returns))
            results[f"var_{name}"] = self.calculate_historical_var(stressed)
            results[f"cvar_{name}"] = self.calculate_cvar(stressed)
        return results

    def assess_risk(self, returns: np.ndarray) -> Dict[str, float]:
        if len(returns) == 0:
            return {"var": 0.0, "cvar": 0.0, "volatility": 0.0, "sharpe": 0.0, "sortino": 0.0, "tail_risk": 0.0}
        volatility = float(np.std(returns, ddof=1))
        downside = returns[returns < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 0 else 0.0
        mean_ret = float(np.mean(returns))
        skewness = float(skew(returns)) if len(returns) > 2 else 0.0
        kurt = float(kurtosis(returns, fisher=True)) if len(returns) > 3 else 0.0
        return {
            "var": self.calculate_historical_var(returns),
            "parametric_var": self.calculate_parametric_var(returns),
            "cornish_fisher_var": self.calculate_cornish_fisher_var(returns),
            "ewma_var": self.calculate_ewma_var(returns),
            "cvar": self.calculate_cvar(returns),
            "expected_shortfall": self.calculate_expected_shortfall(returns),
            "extreme_value_var": self.calculate_extreme_value_var(returns),
            "volatility": volatility,
            "annualized_vol": volatility * np.sqrt(252),
            "sharpe": float(mean_ret / volatility * np.sqrt(252)) if volatility > 0 else 0.0,
            "sortino": float(mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0,
            "tail_risk": self.calculate_tail_risk(returns),
            "skewness": skewness,
            "kurtosis": kurt,
        }