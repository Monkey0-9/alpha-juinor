"""
risk/quantum/tests/test_path_integral.py
"""
import pytest
import numpy as np
from risk.quantum.path_integral import PathIntegralStresser

def test_stress_test_pass():
    stresser = PathIntegralStresser(samples=500, seed=42)

    # Portfolio: 1 Asset, safe
    weights = {"A": 1.0}
    mu = np.array([0.01])
    cov = np.array([[0.0001]]) # Low vol

    # Shock -5%
    result = stresser.stress_test(weights, mu, cov, shock_magnitude=-0.05)

    # Should pass as vol is low, mean shift -5% is not catastrophic (-20% limit)
    assert bool(result.passed) is True
    assert result.stressed_cvar > -0.20

def test_stress_test_fail():
    stresser = PathIntegralStresser(samples=500, seed=42)

    # Portfolio: 1 Asset, highly volatile
    weights = {"A": 1.0}
    mu = np.array([0.0])
    cov = np.array([[0.04]]) # 20% vol

    # Shock -15% -> New Mean -15%.
    # Normal(-15%, 20%) -> 5% tail is roughly -15% - 1.65*20% = -48%
    # This should definitely breach -20% CVaR limit
    result = stresser.stress_test(weights, mu, cov, shock_magnitude=-0.15)

    # result.stressed_cvar should be negative and large magnitude
    assert result.stressed_cvar < -0.20
    assert bool(result.passed) is False

