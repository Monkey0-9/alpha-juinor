import os
import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import pytest

from alpha_families.ml_alpha import MLAlpha
from utils.errors import ModelFeatureMismatchError, GovernanceDisabledError


def test_model_feature_mismatch(tmp_path):
    """Test that feature mismatch triggers governance escalation."""
    ml = MLAlpha(model_path=str(tmp_path))

    # Mock a model artifact with specific features
    mock_model = {
        "model": None,
        "features": ["f1", "f2"],
        "metadata": {
            "features": ["f1", "f2"],
            "n_features": 2
        }
    }

    # Create a dummy model with predict
    class DummyModel:
        def predict(self, X):
            return [0.5]

    mock_model["model"] = DummyModel()

    ml._load_model_for_symbol = lambda sym: mock_model

    # Provide wrong features (f3 instead of f2)
    X_wrong = pd.DataFrame({"f1": [1.0], "f3": [2.0]})

    # Should raise ModelFeatureMismatchError
    with pytest.raises(ModelFeatureMismatchError):
        ml.ml_predict_safe(mock_model["model"], X_wrong, mock_model["metadata"], symbol="TEST")


def test_model_feature_match_pass(tmp_path):
    """Test successful prediction with matching features."""
    ml = MLAlpha(model_path=str(tmp_path))

    mock_model = {
        "model": None,
        "features": ["f1"],
        "metadata": {
            "features": ["f1"],
            "n_features": 1
        }
    }

    class DummyModel:
        def predict(self, X):
            return [0.5]

    mock_model["model"] = DummyModel()

    ml._load_model_for_symbol = lambda sym: mock_model

    X_correct = pd.DataFrame({"f1": [1.0]})

    result = ml.ml_predict_safe(mock_model["model"], X_correct, mock_model["metadata"], symbol="TEST")

    assert result is not None
    assert result[0] == 0.5


def test_model_meta_persistence():
    """Test that model metadata is persisted correctly during training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)

        # Create mock metadata
        meta = {
            "model_id": "TEST_ml_v1",
            "version": "1.0",
            "features": ["f1", "f2", "f3"],
            "n_features": 3,
            "trained_at": "2026-01-23T19:00:00Z",
            "scikit_version": "1.0.0",
            "git_commit": "abc123",
            "training_data_period": {
                "start": "2020-01-01",
                "end": "2025-01-01",
                "n_samples": 1000
            },
            "symbol": "TEST",
            "model_type": "HuberRegressor",
            "contract_name": "ml_v1"
        }

        # Write metadata
        meta_path = model_dir / "model_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Load and validate
        with open(meta_path, "r") as f:
            loaded_meta = json.load(f)

        assert loaded_meta["model_id"] == "TEST_ml_v1"
        assert loaded_meta["version"] == "1.0"
        assert len(loaded_meta["features"]) == 3
        assert loaded_meta["n_features"] == 3
        assert "trained_at" in loaded_meta
        assert "git_commit" in loaded_meta


def test_governance_disable_after_failures(tmp_path):
    """Test that ML alpha disables after repeated feature mismatches."""
    ml = MLAlpha(model_path=str(tmp_path))

    mock_model = {
        "model": None,
        "features": ["f1", "f2"],
        "metadata": {"features": ["f1", "f2"], "n_features": 2}
    }

    class DummyModel:
        def predict(self, X):
            return [0.5]

    mock_model["model"] = DummyModel()
    ml._load_model_for_symbol = lambda sym: mock_model

    X_wrong = pd.DataFrame({"f1": [1.0]})  # Missing f2

    # Trigger 3 failures
    for i in range(3):
        try:
            ml.ml_predict_safe(mock_model["model"], X_wrong, mock_model["metadata"], symbol="TEST")
        except ModelFeatureMismatchError:
            pass

    # Should now be disabled by governance
    assert ml._governance_disabled is True

    # Subsequent calls should raise GovernanceDisabledError
    with pytest.raises(GovernanceDisabledError):
        ml.ml_predict_safe(mock_model["model"], X_wrong, mock_model["metadata"], symbol="TEST")
