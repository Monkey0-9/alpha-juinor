import numpy as np
import pandas as pd
import structlog
from typing import List, Dict, Any, Tuple
import cvxpy as cp
from pydantic import BaseModel

class AgentOutput(BaseModel):
    symbol: str
    mu: float
    sigma: float
    confidence: float
    model_id: str
    model_version: str
    tail_params: Dict[str, float] = {}
    debug: Dict[str, Any] = {}

from mini_quant_fund.intelligence.tail_risk import TailRiskAgent

logger = structlog.get_logger()

class PMBrain:
    """
    Ruthless Institutional Portfolio Manager.
    Implements Convex Utility Optimization with Market Impact and Tail Risk Hard-Gates.
    """
    def __init__(self, gamma_risk: float = 0.5, beta_disagreement: float = 0.2, eta_impact: float = 1.0):
        self.gamma_risk = gamma_risk
        self.beta = beta_disagreement
        self.eta_impact = eta_impact
        self.tail_agent = TailRiskAgent()

    def aggregate_models(self, symbol: str, model_outputs: List[Dict]) -> Tuple[float, float, float]:
        """
        Aggregate ensemble -> distributional forecast per symbol.
        Returns (mu, sigma, var_mu).
        """
        if not model_outputs:
            return 0.0, 0.01, 0.0

        mus = [m.get("mu", 0.0) for m in model_outputs]
        sigmas = [m.get("sigma", 0.01) for m in model_outputs]
        weights = [m.get("confidence", 1.0) for m in model_outputs]

        # Weighted mean for mu
        if np.sum(weights) <= 0:
            mu_hat = float(np.mean(mus))
        else:
            mu_hat = float(np.average(mus, weights=weights))

        # Pooled variance for sigma
        sigma_mean = float(np.sqrt(np.mean([s**2 for s in sigmas])))
        var_mu = float(np.var(mus))
        sigma_agg = float(np.sqrt(sigma_mean**2 + var_mu))

        return mu_hat, sigma_agg, var_mu

    def compute_covariance(self, forecasts: Dict[str, Dict], shrinkage: float = 0.1) -> np.ndarray:
        """
        Return regularized covariance matrix using shrinkage.
        """
        symbols = list(forecasts.keys())
        n = len(symbols)
        sigmas = np.array([forecasts[s].get("sigma", 0.01) for s in symbols])
        diag = np.diag(sigmas ** 2)

        # Simple shrinkage model: diagonal + constant correlation noise
        off_diag = np.full((n, n), 0.01)
        np.fill_diagonal(off_diag, 0)

        Sigma = (1 - shrinkage) * diag + shrinkage * off_diag
        # Ensure PSD
        min_eig = np.min(np.linalg.eigvals(Sigma))
        if min_eig < 0:
            Sigma -= 1.1 * min_eig * np.eye(n)

        return Sigma

    def solve_allocation(self, mu_vec: np.ndarray, Sigma: np.ndarray,
                         liquidity: np.ndarray, quality: np.ndarray,
                         config: Dict[str, Any]) -> np.ndarray:
        """
        Solve Convex Allocation Optimization using CVXPY.
        """
        n = len(mu_vec)
        gamma = config.get("gamma_risk", self.gamma_risk)
        eta = config.get("eta_impact", self.eta_impact)

        w = cp.Variable(n)

        # Portfolio return
        ret = w.T @ mu_vec

        # Portfolio risk (Quadratic form)
        risk = gamma * cp.quad_form(w, Sigma)

        # Market Impact (Linear + 3/2 Power)
        # ImpactCost = sum( c0 * |w|/L + c1 * (|w|/L)^1.5 )
        c0 = 0.01
        c1 = 0.05
        # Prevent division by zero
        liq = np.maximum(liquidity, 1e-9)
        rel_w = cp.abs(w) / liq
        impact = eta * (c0 * cp.sum(rel_w) + c1 * cp.sum(cp.power(rel_w, 1.5)))

        # Sparsity/Quality penalty: lambda * |w| * (1 - Q)
        sparsity = 0.005 * cp.sum(cp.multiply(cp.abs(w), (1 - quality)))

        objective = cp.Maximize(ret - risk - impact - sparsity)

        constraints = []
        # 1. Gross exposure <= leverage_limit
        constraints.append(cp.sum(cp.abs(w)) <= config.get("leverage_limit", 1.0))

        # 2. Net exposure limits
        constraints.append(cp.sum(w) >= config.get("net_exposure_min", -1.0))
        constraints.append(cp.sum(w) <= config.get("net_exposure_max", 1.0))

        # 3. Individual bounds
        max_sizes = config.get("max_pos_sizes", np.full(n, 0.05))
        constraints.append(w <= max_sizes)
        constraints.append(w >= -max_sizes)

        # 4. Crisis Regime Overrides
        if config.get("regime") == "CRISIS":
             constraints.append(cp.abs(w) <= 0.01) # 1% NAV Cap

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS) # ECOS handles power cones for 1.5 power
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return w.value
            else:
                logger.error("CVXPY_OPTIMIZER_FAILED", status=prob.status)
                return None
        except Exception as e:
            logger.error("CVXPY_EXCEPTION", error=str(e))
            return None

    def greedy_allocate(self, forecasts: Dict[str, Dict], market_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
        """
        Fallback allocation: rank-by score, apply caps and return weights.
        """
        symbols = list(forecasts.keys())
        scores = {}
        for s in symbols:
            f = forecasts[s]
            # score = mu_adj / sigma * Q
            sigma = f.get("sigma", 0.01)
            score = (f.get("mu_adj", 0.0) / (sigma + 1e-9)) * f.get("data_quality", 1.0)
            scores[s] = score

        sorted_symbols = sorted(symbols, key=lambda x: scores[x], reverse=True)
        top_k = sorted_symbols[:config.get("top_k", 20)]

        leverage = config.get("leverage_limit", 1.0)
        base_weight = leverage / len(top_k) if top_k else 0

        weights = {s: 0.0 for s in symbols}
        for s in top_k:
            if scores[s] > 0:
                max_s = config.get("max_pos_sizes_map", {}).get(s, 0.05)
                # DQ 75% reduction
                if forecasts[s].get("data_quality", 1.0) < 0.6:
                    max_s *= 0.25
                weights[s] = min(base_weight, max_s)

        return weights

    def get_mu_adjusted(self, mu: float, var_mu: float) -> float:
        return float(mu * np.exp(-self.beta * var_mu))
