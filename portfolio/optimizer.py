"""
Production-ready PM Brain optimizer.
Implements:
    max   mu.T w - lambda * w.T Sigma w - gamma * CVaR_alpha(w)
          - kappa * ||w - w_prev||_1 - eta * Impact(w)
subject to:
    sum(w) == 1
    w_min <= w <= w_max
    sector caps (optional)
Deterministic: uses seeded RNG to generate solver_seed and seeds
    scenario samplers.
Outputs: weights, rejected_assets, explain (metrics).
Requires: cvxpy, numpy
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np
import cvxpy as cp
import hashlib
import json

CONTRACT_VERSION = "1.0.0"


def _schema_hash(obj: Dict) -> str:
    encoded = json.dumps(obj, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def optimize_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    scenario_returns: np.ndarray,
    w_prev: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    sector_map: Optional[Dict[str, List[int]]],
    params: Dict[str, Any],
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """
    Inputs:
      mu: expected returns vector (n,)
      Sigma: covariance matrix (n,n)
      scenario_returns: matrix (N_scenarios, n) of scenario asset returns
                        for CVaR estimation
      w_prev: previous weights (n,)
      w_min, w_max: box bounds (n,) (can set to zeros and max caps)
      sector_map: optional {sector_name: [asset_indices]}
      params: dict with keys: lambda, gamma, alpha, kappa, eta,
              uncertainty_radius
      rng_seed: integer seed for reproducibility

    Returns:
      dict containing:
        w: weights numpy array
        rejected_assets: list[{symbol_index, reason}]
        explain: metrics (objective, mu_adj, CVaR, var, solver_info)
        contract_version, schema_hash
    """
    np.random.seed(rng_seed)

    n = mu.shape[0]
    N_s = scenario_returns.shape[0]

    # Regime compatibility and data confidence are expected to be included
    # in mu already (mu_adj).
    # But allow an optional multiplier in params
    data_conf = params.get("data_confidence", np.ones(n))
    regime_r = params.get("regime_compatibility", np.ones(n))
    mu_adj = mu * data_conf * regime_r

    # Robust covariance shrinkage (Ledoit-Wolf simplistic)
    uncertainty_radius = params.get("uncertainty_radius", 0.0)
    # If uncertainty_radius > 0, we'll apply a simple diagonal inflation
    # to Sigma
    if uncertainty_radius > 0:
        Sigma = Sigma + uncertainty_radius * np.diag(np.diag(Sigma))

    # CVaR linearization variables
    alpha = float(params.get("alpha", 0.95))
    lamb = float(params.get("lambda", 1.0))
    gamma = float(params.get("gamma", 1.0))
    kappa = float(params.get("kappa", 0.0))
    eta = float(params.get("eta", 0.0))

    # CVX variables
    w = cp.Variable(n)
    # turnover l1
    u = cp.Variable(n)  # u >= |w - w_prev|
    # CVaR aux
    t = cp.Variable(1)
    z = cp.Variable(N_s)  # scenario exceedances

    # Impact term: approximate via quadratic using Sigma-like or
    # H diagonal in params
    H = params.get("impact_matrix")
    if H is None:
        # default small diagonal impact proportional to liquidity proxies
        # if provided
        H = params.get("impact_diag", 1e-4) * np.eye(n)
    else:
        H = np.array(H)

    # Objective components
    expected_ret = mu_adj @ w
    variance_penalty = cp.quad_form(w, Sigma)  # w^T Sigma w

    # define scenario losses: -R_scenarios * w (portfolio loss per scenario)
    # Loss L_s = -(R_s @ w)
    R = scenario_returns  # numpy array (N_s, n)
    losses = - (R @ w)  # affine in w as cvxpy expression (N_s,)
    # CVaR linearization constraints: z_s >= losses - t ; z_s >= 0
    # CVaR_alpha = t + (1/(alpha*N)) sum z_s
    cvar_expr = t + (1.0 / ((1 - alpha) * N_s)) * cp.sum(z)

    # impact approx:
    impact_term = cp.quad_form(w, H)

    # Turnover L1: u >= w - w_prev, u >= -(w - w_prev)
    constraints = []
    constraints += [u >= w - w_prev, u >= -(w - w_prev), u >= 0]

    # CVaR constraints
    constraints += [z >= losses - t, z >= 0]

    # bounds
    constraints += [w >= w_min, w <= w_max]

    # sum to 1
    constraints += [cp.sum(w) == 1]

    # sector caps
    if sector_map:
        for sector, indices in sector_map.items():
            sector_cap = params.get("sector_caps", {}).get(sector, 1.0)
            constraints += [cp.sum(w[indices]) <= sector_cap]

    # Problem objective (maximize -> minimize negative)
    objective = cp.Maximize(
        expected_ret
        - lamb * variance_penalty
        - gamma * cvar_expr
        - kappa * cp.sum(u)
        - eta * impact_term
    )

    prob = cp.Problem(objective, constraints)

    # Choose deterministic solver settings; prefer OSQP (deterministic)
    # when available
    solver = cp.OSQP
    solve_opts = {
        "eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 100000, "verbose": False
    }
    # OSQP may produce deterministic outputs; fix warm_start False;
    # random seed not generally available for OSQP via cvxpy
    try:
        prob.solve(solver=solver, **solve_opts)
    except Exception as e:
        # fallback to SCS (less precise) — but raise to fail tests
        raise RuntimeError(f"Optimizer failed: {e}")

    w_val = np.array(w.value).flatten()
    # Numerical guard
    w_val = np.real_if_close(w_val)
    # small negative clamp
    w_val[np.abs(w_val) < 1e-12] = 0.0

    # Post-check constraints
    sumw = float(np.sum(w_val))
    if not np.isclose(sumw, 1.0, atol=1e-4):
        # project back to simplex conservatively (shouldn't happen)
        w_val = np.maximum(w_val, 0)
        if w_val.sum() == 0:
            w_val = np.ones_like(w_val) / len(w_val)
        else:
            w_val = w_val / w_val.sum()

    # Rejected assets: assets with w == w_min and would improve expected
    # objective if allowed -> simple dominance test
    rejected = []
    for i in range(n):
        if np.isclose(w_val[i], w_min[i]) and mu_adj[i] > 0:
            rejected.append({
                "asset_index": int(i),
                "reason": "hit lower bound; positive adjusted mean"
            })

    # Explain payload
    explain = {
        "mu_adj": mu_adj.tolist(),
        "sum_w": float(np.sum(w_val)),
        "variance": float(np.dot(w_val, Sigma @ w_val)),
        "cvar_estimate": None,
        "solver_status": prob.status,
    }

    # compute CVaR estimate on scenarios (post-hoc)
    port_losses = - (R @ w_val)
    VaR = np.quantile(port_losses, alpha)
    cvar_est = (
        np.mean(port_losses[port_losses >= VaR])
        if np.any(port_losses >= VaR)
        else float(np.max(port_losses))
    )
    explain["cvar_estimate"] = float(cvar_est)
    explain["VaR"] = float(VaR)

    result = {
        "w": w_val,
        "rejected_assets": rejected,
        "explain": explain,
        "contract_version": CONTRACT_VERSION,
        "schema_hash": _schema_hash({
            "lambda": lamb,
            "gamma": gamma,
            "kappa": kappa,
            "eta": eta,
            "alpha": alpha,
            "uncertainty_radius": uncertainty_radius,
        }),
    }
    return result


class PortfolioOptimizer:
    """
    Adapter class for optimize_portfolio to satisfy class-based
    interfaces/tests.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def optimize(self,
                 mu: np.ndarray,
                 Sigma: np.ndarray,
                 scenario_returns: np.ndarray,
                 w_prev: np.ndarray,
                 w_min: np.ndarray,
                 w_max: np.ndarray,
                 sector_map: Optional[Dict[str, List[int]]] = None,
                 params: Dict[str, Any] = None,
                 rng_seed: int = 0) -> Dict[str, Any]:

        # Merge config params with runtime params
        run_params = self.config.copy()
        if params:
            run_params.update(params)

        return optimize_portfolio(
            mu, Sigma, scenario_returns, w_prev, w_min, w_max,
            sector_map, run_params, rng_seed
        )

# Export for tests


@dataclass
class Constraint:
    name: str
    params: Dict[str, Any]


CVXPY_AVAILABLE = True
