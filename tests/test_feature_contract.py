"""
tests/test_feature_contract.py
Unit tests for feature contract loader and validation
"""
import pytest
import json
from pathlib import Path

from features.contract import (
    load_feature_contract,
    get_feature_list,
    validate_contract_compliance,
    FeatureContractError
)


def test_load_feature_contract():
    """Test loading the ML v1 feature contract."""
    contract = load_feature_contract("ml_v1")

    assert contract is not None
    assert "name" in contract
    assert "version" in contract
    assert "features" in contract
    assert "n_features" in contract

    assert contract["name"] == "ml_v1"
    assert contract["n_features"] == 28
    assert len(contract["features"]) == 28


def test_get_feature_list():
    """Test getting ordered feature list from contract."""
    features = get_feature_list("ml_v1")

    assert len(features) == 28
    assert isinstance(features, list)
    assert all(isinstance(f, str) for f in features)

    # Check first few features
    assert features[0] == "ret_1d"
    assert features[1] == "ret_5d"
    assert "rsi_14" in features
    assert "macd" in features


def test_validate_compliance_success():
    """Test successful contract compliance validation."""
    contract_features = get_feature_list("ml_v1")

    result = validate_contract_compliance(
        contract_features,
        contract_name="ml_v1",
        strict_order=True
    )

    assert result["compliant"] is True
    assert len(result["missing"]) == 0
    assert len(result["extra"]) == 0
    assert result["order_mismatch"] is False


def test_validate_compliance_missing_features():
    """Test validation failure when features are missing."""
    incomplete_features = ["ret_1d", "ret_5d", "vol_5d"]  # Only 3 features

    result = validate_contract_compliance(
        incomplete_features,
        contract_name="ml_v1",
        strict_order=False
    )

    assert result["compliant"] is False
    assert len(result["missing"]) > 0
    assert "macd" in result["missing"]
    assert "rsi_14" in result["missing"]


def test_validate_compliance_extra_features():
    """Test validation failure when extra features present."""
    contract_features = get_feature_list("ml_v1")
    extra_features = contract_features + ["unknown_feature", "another_extra"]

    result = validate_contract_compliance(
        extra_features,
        contract_name="ml_v1",
        strict_order=False
    )

    assert result["compliant"] is False
    assert len(result["extra"]) == 2
    assert "unknown_feature" in result["extra"]


def test_validate_compliance_order_mismatch():
    """Test validation failure when feature order is wrong."""
    contract = load_feature_contract("ml_v1")
    reordered_features = sorted(contract["features"])  # Alphabetically sorted

    result = validate_contract_compliance(
        reordered_features,
        contract_name="ml_v1",
        strict_order=True
    )

    # Should fail due to order mismatch (even though all features present)
    assert result["order_mismatch"] is True
    assert result["compliant"] is False


def test_load_nonexistent_contract():
    """Test error handling when contract doesn't exist."""
    with pytest.raises(FeatureContractError, match="Feature contract not found"):
        load_feature_contract("nonexistent_contract")


def test_contract_schema_structure():
    """Test that contract has all required metadata fields."""
    contract = load_feature_contract("ml_v1")

    assert "description" in contract
    assert "created_at" in contract
    assert "dtype" in contract
    assert "enforce_order" in contract
    assert "enforce_dtype" in contract

    assert contract["dtype"] == "float32"
    assert contract["enforce_order"] is True
    assert contract["enforce_dtype"] is True
