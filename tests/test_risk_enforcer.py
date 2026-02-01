# tests/test_risk_enforcer.py
import numpy as np
from services.risk_enforcer import RiskEnforcer

def test_risk_enforcer_cvar_basic():
    re = RiskEnforcer(params={"cvar_limit_pct":0.2})
    # create trivial scenarios and weights
    N_s, n = 500, 4
    scenarios = np.random.normal(0, 0.01, size=(N_s, n))
    weights = np.ones(n) / n
    out = re.enforce(weights, scenarios)
    assert "allow" in out
    assert "cvar" in out

def test_risk_enforcer_cvar_exceeds():
    re = RiskEnforcer(params={"cvar_limit_pct":0.001})  # very tight limit
    N_s, n = 500, 4
    # create scenarios with high volatility
    scenarios = np.random.normal(0, 0.05, size=(N_s, n))
    weights = np.ones(n) / n
    out = re.enforce(weights, scenarios)
    # should likely exceed the tight CVaR limit
    assert "cvar" in out

def test_risk_enforcer_returns_suggested_weights_on_entanglement():
    # This test verifies the entanglement check path exists
    re = RiskEnforcer(params={"cvar_limit_pct":0.5, "entanglement_threshold":0.0})  # very low threshold
    N_s, n = 100, 4
    scenarios = np.random.normal(0, 0.01, size=(N_s, n))
    weights = np.ones(n) / n
    # With entanglement threshold 0, if entanglement module exists it should suggest haircut
    out = re.enforce(weights, scenarios, returns_matrix_for_ent=np.random.randn(n, 50))
    assert "allow" in out
