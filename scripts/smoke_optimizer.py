import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.portfolio.test_optimizer_deterministic import make_toy_problem
from portfolio.optimizer import optimize_portfolio
import numpy as np

def run_smoke():
    try:
        mu, Sigma, scenarios, w_prev, w_min, w_max, sector_map, params = make_toy_problem()
        res = optimize_portfolio(mu, Sigma, scenarios, w_prev, w_min, w_max, sector_map, params, rng_seed=1000)

        print(f"sum w: {res['explain']['sum_w']}")
        print(f"cvar: {res['explain']['cvar_estimate']}")
        print(f"schema_hash: {res['schema_hash']}")
        print(f"Solver Status: {res['explain']['solver_status']}")

        # Additional checks
        w = res['w']
        if np.abs(np.sum(w) - 1.0) > 1e-4:
            print("FAIL: Sum weights != 1.0")
            exit(1)
        if np.any(w < w_min - 1e-8) or np.any(w > w_max + 1e-8):
            print("FAIL: Bounds violation")
            exit(1)

        print("Smoke Test Passed")

    except Exception as e:
        print(f"Smoke Test Failed: {e}")
        exit(1)

if __name__ == "__main__":
    run_smoke()
