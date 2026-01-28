"""
tests/test_training_inference_parity.py
P1-2: Training-Inference Parity Tests
"""
import pytest
import json
from pathlib import Path

from features.contract import load_feature_contract
from data.processors.features import compute_features_for_symbol
import pandas as pd
import numpy as np


def test_training_inference_parity_ml_v1():
    """
    Test that training feature order matches runtime inference.

    This is critical - if features are in different order at training vs inference,
    model predictions will be completely wrong.
    """
    # Load feature contract (used during training)
    contract = load_feature_contract("ml_v1")
    training_features = contract["features"]

    # Generate runtime features
    # Create synthetic OHLCV data
    dates = pd.date_range(end='2026-01-01', periods=300)
    df = pd.DataFrame({
        'Open': np.random.randn(300) * 10 + 100,
        'High': np.random.randn(300) * 10 + 105,
        'Low': np.random.randn(300) * 10 + 95,
        'Close': np.random.randn(300) * 10 + 100,
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)

    runtime_features_df = compute_features_for_symbol(df, contract_name="ml_v1")
    runtime_features = list(runtime_features_df.columns)

    # CRITICAL: Feature order must match exactly
    assert training_features == runtime_features, (
        f"PARITY FAILURE!\n"
        f"Training features: {training_features[:5]}...\n"
        f"Runtime features:  {runtime_features[:5]}...\n"
        f"This will cause incorrect model predictions!"
    )


def test_feature_count_matches():
    """Test that feature count is consistent."""
    contract = load_feature_contract("ml_v1")

    assert contract["n_features"] == 28
    assert len(contract["features"]) == 28


def test_model_metadata_exists_and_matches():
    """Test that saved model metadata matches contract."""
    # Check if any models exist
    model_dir = Path("models/ml_alpha")

    if not model_dir.exists():
        pytest.skip("No models found - run training first")

    # Find model metadata files
    meta_files = list(model_dir.glob("*/model_meta.json"))

    if not meta_files:
        pytest.skip("No model metadata found - models need retraining")

    # Check first model
    with open(meta_files[0], 'r') as f:
        model_meta = json.load(f)

    contract = load_feature_contract("ml_v1")
    contract_features = contract["features"]
    model_features = model_meta.get("features", [])

    # Parity check
    assert model_features == contract_features, (
        f"Model was trained with different features!\n"
        f"Model has: {len(model_features)} features\n"
        f"Contract expects: {len(contract_features)} features\n"
        f"Model must be retrained!"
    )
