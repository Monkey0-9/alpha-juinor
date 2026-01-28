"""
risk/quantum/tests/test_state_space.py
"""
import pytest
import numpy as np
from risk.quantum.state_space import RegimeStateSpace
from risk.quantum.contracts import QuantumState

def test_regime_initialization():
    rss = RegimeStateSpace(n_regimes=5)
    assert len(rss.belief) == 5
    assert np.isclose(np.sum(rss.belief), 1.0)

def test_regime_update():
    rss = RegimeStateSpace()
    state = rss.update()

    assert isinstance(state, QuantumState)
    assert len(state.regime_belief) == 5
    assert np.isclose(sum(state.regime_belief), 1.0)
    assert state.contract_version is not None

def test_compatibility():
    rss = RegimeStateSpace(n_regimes=3)
    # Force belief
    rss.belief = np.array([1.0, 0.0, 0.0])

    # Profile matches regime 0 perfectly
    profile_good = [1.0, 0.0, 0.0]
    score = rss.get_compatibility(profile_good)
    assert score == 1.0

    # Profile matches regime 1 (curr prob 0)
    profile_bad = [0.0, 1.0, 0.0]
    score = rss.get_compatibility(profile_bad)
    assert score == 0.0
