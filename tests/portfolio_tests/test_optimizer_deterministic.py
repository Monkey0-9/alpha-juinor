
import numpy as np
from portfolio.optimizer import optimize_portfolio
import pytest

def make_toy_problem(seed=123):
    np.random.seed(seed)
    n = 6
    mu = np.random.normal(0.001, 0.01, size=n)
    A = np.random.randn(n, n)
    Sigma = np.cov(A) + np.eye(n) * 1e-3
    # create scenario returns (500 scenarios)
    scenarios = np.random.normal(loc=mu, scale=np.sqrt(np.diag(Sigma)), size=(500, n))
    w_prev = np.ones(n) / n
    w_min = np.zeros(n)
    w_max = np.ones(n) * 0.4
    sector_map = None
    params = {
        "lambda": 2.0,
        "gamma": 6.0,
        "alpha": 0.95,
        "kappa": 0.02,
        "eta": 1e-4,
        "uncertainty_radius": 0.0,
        "data_confidence": np.ones(n),
        "regime_compatibility": np.ones(n),
    }
    return mu, Sigma, scenarios, w_prev, w_min, w_max, sector_map, params

def test_optimizer_deterministic():
    mu, Sigma, scenarios, w_prev, w_min, w_max, sector_map, params = make_toy_problem(seed=42)
    r1 = optimize_portfolio(mu, Sigma, scenarios, w_prev, w_min, w_max, sector_map, params, rng_seed=1000)
    r2 = optimize_portfolio(mu, Sigma, scenarios, w_prev, w_min, w_max, sector_map, params, rng_seed=1000)
    # assert allocations identical within tight tolerance
    w1 = r1["w"]
    w2 = r2["w"]
    assert np.allclose(w1, w2, atol=1e-6), f"Determinism failed: {w1} vs {w2}"
    # basic feasibility checks
    assert np.isclose(np.sum(w1), 1.0, atol=1e-4)
    assert np.all(w1 >= w_min - 1e-8)
    assert np.all(w1 <= w_max + 1e-8)
