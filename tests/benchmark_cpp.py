import os
import sys
import time
import numpy as np

# Setup path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "nexus", "cpp_extensions")
    )
)

# Add MinGW bin to DLL search path for Python 3.8+
mingw_bin = os.path.expanduser(r"~\scoop\apps\mingw\current\bin")
if os.path.exists(mingw_bin):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(mingw_bin)
    else:
        os.environ["PATH"] = mingw_bin + os.pathsep + os.environ["PATH"]

import nexus_cpp  # noqa: E402


def pure_python_survival(
    initial_capital, mu, sigma, days, n_simulations, ruin_threshold
):
    if sigma == 0:
        return 1.0
    ruin_level = initial_capital * (1 - ruin_threshold)
    survived = 0
    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        path_returns = rng.normal(mu, sigma, days)
        prices = initial_capital * np.cumprod(1 + path_returns)
        if np.min(prices) > ruin_level:
            survived += 1
    return survived / n_simulations


def run_benchmark():
    initial_capital = 100000.0
    mu = 0.0005
    sigma = 0.015
    days = 252
    n_simulations = 100000  # Large number to show speedup
    ruin_threshold = 0.5

    print(f"Running {n_simulations} simulations...")

    # 1. C++ Benchmark
    start_cpp = time.time()
    surv_cpp = nexus_cpp.run_survival_analysis(
        initial_capital, mu, sigma, days, n_simulations, ruin_threshold
    )
    end_cpp = time.time()

    # 2. Python Benchmark
    start_py = time.time()
    surv_py = pure_python_survival(
        initial_capital, mu, sigma, days, n_simulations, ruin_threshold
    )
    end_py = time.time()

    print("\n--- Results ---")
    print(f"C++ Survival Probability: {surv_cpp:.4f}")
    print(f"Python Survival Probability: {surv_py:.4f}")
    print("\n--- Performance ---")
    print(f"C++ Execution Time: {end_cpp - start_cpp:.4f} seconds")
    print(f"Python Execution Time: {end_py - start_py:.4f} seconds")
    print(f"Speedup: {(end_py - start_py) / (end_cpp - start_cpp):.2f}x faster")


if __name__ == "__main__":
    run_benchmark()
