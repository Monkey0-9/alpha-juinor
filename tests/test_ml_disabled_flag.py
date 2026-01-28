"""
tests/test_ml_disabled_flag.py
Unit tests for ML alpha config-based enable/disable functionality
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch

from alpha_families.ml_alpha import MLAlpha


def test_ml_alpha_disabled_by_config(tmp_path):
    """Test that ML alpha respects ml_enabled=false config flag."""
    ml = MLAlpha(model_path=str(tmp_path))

    # Mock config to have ml_enabled=false
    mock_config = {
        "features": {
            "ml_enabled": False
        }
    }

    with patch('configs.config_manager.ConfigManager') as mock_cm:
        mock_cm.return_value.config = mock_config

        result = ml.generate_signal(pd.DataFrame(), symbol="TEST")

        assert result["signal"] == 0.0
        assert result["confidence"] == 0.0
        assert result["metadata"]["reason"] == "ML_DISABLED_BY_CONFIG"


def test_ml_alpha_enabled_by_config(tmp_path):
    """Test that ML alpha attempts prediction when ml_enabled=true."""
    ml = MLAlpha(model_path=str(tmp_path))

    # Mock config to have ml_enabled=true
    mock_config = {
        "features": {
            "ml_enabled": True
        }
    }

    with patch('configs.config_manager.ConfigManager') as mock_cm:
        mock_cm.return_value.config = mock_config

        result = ml.generate_signal(pd.DataFrame(), symbol="TEST")

        # Should not return ML_DISABLED_BY_CONFIG
        # (may return other reasons like MODEL_MISSING or INSUFFICIENT_DATA)
        assert result["metadata"].get("reason") != "ML_DISABLED_BY_CONFIG"


def test_ml_alpha_no_model_loading_when_disabled(tmp_path, monkeypatch):
    """Test that model loading is not attempted when ML disabled."""
    ml = MLAlpha(model_path=str(tmp_path))

    load_called = []

    def mock_load(symbol):
        load_called.append(symbol)
        return None

    monkeypatch.setattr(ml, "_load_model_for_symbol", mock_load)

    # Mock config to have ml_enabled=false
    mock_config = {
        "features": {
            "ml_enabled": False
        }
    }

    with patch('configs.config_manager.ConfigManager') as mock_cm:
        mock_cm.return_value.config = mock_config

        ml.generate_signal(pd.DataFrame(), symbol="TEST")

        # Model loader should NOT have been called
        assert len(load_called) == 0


def test_governance_disabled_overrides_config(tmp_path):
    """Test that governance disable takes precedence even if config enables ML."""
    ml = MLAlpha(model_path=str(tmp_path))

    # Force governance disabled state
    ml._governance_disabled = True

    # Mock config to enable ML
    mock_config = {
        "features": {
            "ml_enabled": True
        }
    }

    with patch('configs.config_manager.ConfigManager') as mock_cm:
        mock_cm.return_value.config = mock_config

        result = ml.generate_signal(pd.DataFrame(), symbol="TEST")

        assert result["signal"] == 0.0
        assert result["confidence"] == 0.0
        assert result["metadata"]["reason"] == "ML_DISABLED_BY_GOVERNANCE"
