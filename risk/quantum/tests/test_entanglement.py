"""
risk/quantum/tests/test_entanglement.py
"""
import pytest
import pandas as pd
import numpy as np
from risk.quantum.entanglement_detector import EntanglementDetector

def test_entanglement_metric():
    # 1. Perfectly correlated data (Entangled)
    x = np.random.normal(0, 1, 100)
    data = pd.DataFrame({
        "A": x,
        "B": x * 0.99 + 0.01, # High corr
        "C": x * 0.95 + 0.05
    })

    det = EntanglementDetector(threshold=0.5)
    report = det.compute_metric(data)

    # Global index should be high (near 1.0 for perfect sync)
    assert report.global_index > 0.8
    assert bool(report.threshold_breach) is True

def test_uncorrelated_noise():
    # 2. Random noise (Unentangled)
    np.random.seed(42)
    data = pd.DataFrame(np.random.normal(0, 1, (100, 5)), columns=list("ABCDE"))

    det = EntanglementDetector(threshold=0.8)
    report = det.compute_metric(data)

    # Random metrics usually low (~0.2-0.4 for small N)
    assert report.global_index < 0.6
    assert bool(report.threshold_breach) is False
    assert len(report.asset_centrality) == 5

