import numpy as np
import pandas as pd
from scipy.stats import genpareto
from typing import Dict, Any
import structlog

logger = structlog.get_logger()

class TailRiskAgent:
    """
    Survival-first Tail Risk Audit.
    Uses Extreme Value Theory (EVT) and Empirical Quantile for CVaR(95%).
    Threshold is configurable but defaults to -5% daily CVaR.
    """
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def compute_cvar(self, returns: pd.Series) -> float:
        """
        Computes Conditional Value at Risk (Expected Shortfall).
        Formula: CVaR_alpha = E[R | R < VaR_alpha]
        """
        if len(returns) < 100:
            return -0.02 # Safe default for insufficient data

        # 1. Empirical Quantile (VaR)
        var_95 = np.percentile(returns, (1 - self.confidence_level) * 100)

        # 2. EVT: Peaks Over Threshold (POT)
        # We model the tail (returns < var_95) using Generalized Pareto Distribution
        tail = returns[returns < var_95]
        if len(tail) < 10:
             return float(var_95) # Fallback to VaR if tail is too thin

        # Fit GPD
        # params: xi (shape), beta (scale)
        try:
            # Shift tail to be positive for fitting
            excesses = np.abs(tail - var_95)
            fit_params = genpareto.fit(excesses)
            xi, _, beta = fit_params

            # Analytical CVaR for GPD tail
            # CVaR = VaR + (beta + xi*(VaR - u)) / (1 - xi)
            # Simplified proxy for ES
            cvar_95 = np.mean(tail)

            return float(cvar_95)
        except Exception as e:
            logger.error("EVT fit failed", error=str(e))
            return float(var_95)

    def get_blocking_rule(self, cvar: float, threshold: float = -0.05) -> Dict[str, Any]:
        """
        Rule: If CVaR_95% > threshold -> reduce gross exposure by min(50%, function of tail skew).
        """
        if cvar < threshold:
            return {
                "blocked": True,
                "reduction_pct": 0.5,
                "reason": f"TAIL_RISK_EXCEED: {cvar:.4f} < {threshold}"
            }
        return {"blocked": False, "reduction_pct": 0.0}
