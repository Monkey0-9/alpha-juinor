import logging
import numpy as np
from scipy.stats import norm
from typing import Dict

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Institutional Risk Engine for the Nexus Platform.
    Calculates Value-at-Risk (VaR), CVaR, and performs stress testing.
    """
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_var(self, returns: np.ndarray) -> float:
        """
        Calculates Parametric Value-at-Risk (VaR).
        """
        if len(returns) == 0:
            return 0.0
        mu = np.mean(returns)
        sigma = np.std(returns)
        var = norm.ppf(1 - self.confidence_level, mu, sigma)
        return float(var)

    def calculate_cvar(self, returns: np.ndarray) -> float:
        """
        Calculates Conditional Value-at-Risk (CVaR / Expected Shortfall).
        """
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(returns)
        mask = returns <= var
        if not np.any(mask):
            return var
        expected_loss = float(np.mean(returns[mask]))
        return expected_loss

    def assess_risk(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive risk assessment returning VaR, CVaR, and other metrics.
        """
        if len(returns) == 0:
            return {"var": 0.0, "cvar": 0.0, "volatility": 0.0, "sharpe": 0.0}

        var = self.calculate_var(returns)
        cvar = self.calculate_cvar(returns)
        volatility = float(np.std(returns))
        sharpe = float(np.mean(returns) / volatility) if volatility > 0 else 0.0

        return {
            "var": var,
            "cvar": cvar,
            "volatility": volatility,
            "sharpe": sharpe
        }
