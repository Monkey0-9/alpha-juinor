"""
tests/test_ml_mode_shadow.py
P0-2: Test that shadow mode logs predictions but doesn't trade
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import json

from alpha_families.ml_alpha import MLAlpha


def test_ml_mode_disabled():
    """Test that disabled mode returns neutral signals."""
    ml = MLAlpha()
    ml.ml_mode = 'disabled'

    result = ml.generate_signal(pd.DataFrame(), symbol="TEST")

    assert result["signal"] == 0.0
    assert result["confidence"] == 0.0
    assert result["metadata"]["reason"] == "ML_MODE_DISABLED"


def test_ml_mode_shadow_returns_neutral():
    """Test that shadow mode returns neutral signal even when model predicts."""
    ml = MLAlpha()
    ml.ml_mode = 'shadow'

    # Mock model that returns 0.5
    mock_model_artifact = {
        "model": MagicMock(predict=lambda x: [0.5]),
        "metadata": {"features": ["f1"], "n_features": 1}
    }

    ml._load_model_for_symbol = lambda sym: mock_model_artifact
    ml._extract_features = lambda df: pd.DataFrame({"f1": [1.0]})

    with patch('configs.config_manager.ConfigManager') as mock_cm:
        mock_cm.return_value.config = {"features": {"ml_enabled": True, "ml_mode": "shadow"}}

        result = ml.generate_signal(pd.DataFrame(), symbol="TEST")

        # Shadow mode returns neutral
        assert result["signal"] == 0.0
        assert result["confidence"] == 0.0
        assert result["metadata"]["status"] == "SHADOW"
        # But logs actual prediction
        assert "shadow_signal" in result["metadata"]
        assert result["metadata"]["shadow_signal"] != 0.0


def test_ml_mode_live_requires_model():
    """Test that live mode raises error if model missing."""
    ml = MLAlpha()
    ml.ml_mode = 'live'

    ml._load_model_for_symbol = lambda sym: None  # No model

    with patch('configs.config_manager.ConfigManager') as mock_cm:
        mock_cm.return_value.config = {"features": {"ml_enabled": True, "ml_mode": "live"}}

        with pytest.raises(ValueError, match="LIVE mode requires model"):
            ml.generate_signal(pd.DataFrame(), symbol="TEST")


def test_ml_mode_live_returns_signal():
    """Test that live mode returns actual predictions."""
    ml = MLAlpha()
    ml.ml_mode = 'live'

    mock_model_artifact = {
        "model": MagicMock(predict=lambda x: [0.7]),
        "metadata": {"features": ["f1"], "n_features": 1}
    }

    ml._load_model_for_symbol = lambda sym: mock_model_artifact
    ml._extract_features = lambda df: pd.DataFrame({"f1": [1.0]})

    with patch('configs.config_manager.ConfigManager') as mock_cm:
        mock_cm.return_value.config = {"features": {"ml_enabled": True, "ml_mode": "live"}}

        result = ml.generate_signal(pd.DataFrame(), symbol="TEST")

        assert result["signal"] != 0.0  # Live mode returns real signal
        assert result["metadata"]["ml_mode"] == "live"


def test_shadow_mode_logging(caplog):
    """Test that shadow mode logs predictions."""
    ml = MLAlpha()
    ml.ml_mode = 'shadow'

    mock_model_artifact = {
        "model": MagicMock(predict=lambda x: [0.6]),
        "metadata": {"features": ["f1"], "n_features": 1}
    }

    ml._load_model_for_symbol = lambda sym: mock_model_artifact
    ml._extract_features = lambda df: pd.DataFrame({"f1": [1.0]})

    with patch('configs.config_manager.ConfigManager') as mock_cm:
        mock_cm.return_value.config = {"features": {"ml_enabled": True, "ml_mode": "shadow"}}

        with caplog.at_level('INFO'):
            result = ml.generate_signal(pd.DataFrame(), symbol="TEST")

            # Check that shadow prediction was logged
            shadow_logs = [r for r in caplog.records if 'ML_SHADOW' in r.getMessage()]
            assert len(shadow_logs) > 0

            # Parse JSON log
            log_msg = shadow_logs[0].getMessage()
            log_data = json.loads(log_msg)
            assert log_data["component"] == "ML_SHADOW"
            assert log_data["symbol"] == "TEST"
            assert "prediction" in log_data
            assert "features_hash" in log_data
