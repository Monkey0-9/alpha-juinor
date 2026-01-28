# tests/test_helpers.py
"""
Test helper utilities for generating sample inputs.
"""
import time
import numpy as np


def sample_input(run_id="run_example", n=6, T=200):
    """
    Generate a sample input for testing the DecisionAgent.

    Args:
        run_id: Unique run identifier
        n: Number of assets for returns matrix
        T: Number of time periods

    Returns:
        Dict with full input schema
    """
    np.random.seed(0)
    price_series = (100 + np.cumsum(np.random.normal(0, 0.5, size=T))).tolist()
    returns = np.diff(np.log(price_series)).tolist()
    features = {
        "rsi_3": 18.0,
        "boll_z": -2.4,
        "ema_9": 99.0,
        "ema_21": 102.5,
        "atr_pct": 1.8,
        "macd_hist": -0.2,
        "volume_z": 2.0,
        "adv_ok": True
    }
    inp = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "symbol": "TEST",
        "price": price_series[-1],
        "nav_usd": 1_000_000,
        "features": features,
        "historical": {
            "price_series": price_series,
            "returns_matrix": [returns for _ in range(n)]
        },
        "position_state": {"has_position": False, "qty": 0},
        "execution": {"slippage_bps": 6.0, "spread_bps": 3.0, "adv_usd": 2_000_000},
        "models": {"mean_reversion_score": 0.82},
        "risk": {"stop_distance_pct": 0.02, "cvar_limit": 0.05},
        "data_confidence": 0.95
    }
    return inp
