# tests/test_institutional_stability.py
import pytest
import pandas as pd
import numpy as np
from portfolio.pm_brain import pm_brain_run
from ml.validate_features import validate_feature_schema, REQUIRED_FEATURES
from utils.metrics import metrics
from utils.alerts import check_system_thresholds

def test_pm_brain_filtering():
    """Test that PM Brain selects only top-K and respects impact."""
    candidates = [
        {"symbol": "AAPL", "mu": 0.05, "sigma": 0.1, "data_conf": 1.0, "liquidity": 1.0, "impact_bps": 5.0},
        {"symbol": "NVDA", "mu": 0.06, "sigma": 0.15, "data_conf": 1.0, "liquidity": 1.0, "impact_bps": 5.0},
        {"symbol": "MSFT", "mu": 0.04, "sigma": 0.1, "data_conf": 1.0, "liquidity": 1.0, "impact_bps": 5.0},
        {"symbol": "TSLA", "mu": 0.10, "sigma": 0.5, "data_conf": 1.0, "liquidity": 1.0, "impact_bps": 50.0}, # High impact
    ]

    selected = pm_brain_run(candidates, 1000000, policy={"max_trades_per_day": 2, "impact_limit_bps": 20.0, "data_conf_threshold": 0.5, "liquidity_threshold": 0.5, "max_position_pct": 0.1, "min_position_pct": 0.01, "max_gross_exposure": 1.0, "kelly_gamma": 0.15})

    assert len(selected) == 2
    symbols = [s["symbol"] for s in selected]
    assert "TSLA" not in symbols # Impact gate
    assert "NVDA" in symbols # Top mu/sigma

def test_feature_validator():
    """Test that feature validator catches missing institutional features."""
    df = pd.DataFrame(columns=["returns_1d", "volatility_20d"])
    is_valid, missing = validate_feature_schema(df, REQUIRED_FEATURES)
    assert not is_valid
    assert len(missing) > 0
    assert "returns_5d" in missing

def test_alerts_escalation(caplog):
    """Test that metrics trigger alerts."""
    metrics.reset("ml_feature_mismatch_total")
    metrics.inc("ml_feature_mismatch_total", 1)

    with caplog.at_level("CRITICAL"):
        check_system_thresholds()
        assert "PAGER_DUTY" in caplog.text
        assert "ML Feature Schema Mismatch" in caplog.text

if __name__ == "__main__":
    pytest.main([__file__])
