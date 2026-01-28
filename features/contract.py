"""
features/contract.py
Feature Contract Loader - Single Source of Truth for ML Features
"""
import json
from pathlib import Path
from typing import Dict, List, Any


class FeatureContractError(Exception):
    """Raised when feature contract is invalid or missing."""
    pass


def load_feature_contract(name: str = "ml_v1") -> Dict[str, Any]:
    """
    Load feature contract from JSON file.

    Args:
        name: Contract name (e.g., "ml_v1")

    Returns:
        Dict containing contract specification with keys:
            - name: str
            - version: str
            - features: List[str] (ordered list of feature names)
            - n_features: int
            - dtype: str
            - enforce_order: bool
            - enforce_dtype: bool

    Raises:
        FeatureContractError: If contract file missing or invalid

    Example:
        >>> contract = load_feature_contract("ml_v1")
        >>> print(contract["features"])
        ['ret_1d', 'ret_5d', ...]
        >>> print(contract["n_features"])
        28
    """
    contract_path = Path(__file__).parent / "feature_contracts" / f"feature_contract_{name}.json"

    if not contract_path.exists():
        raise FeatureContractError(
            f"Feature contract not found: {contract_path}\n"
            f"Expected contract file: feature_contract_{name}.json"
        )

    try:
        with open(contract_path, "r") as f:
            contract = json.load(f)
    except json.JSONDecodeError as e:
        raise FeatureContractError(f"Invalid JSON in feature contract {name}: {e}")

    # Validate contract structure
    required_fields = ["name", "version", "features", "n_features"]
    missing_fields = [field for field in required_fields if field not in contract]
    if missing_fields:
        raise FeatureContractError(
            f"Feature contract {name} missing required fields: {missing_fields}"
        )

    # Validate features list
    if not isinstance(contract["features"], list):
        raise FeatureContractError(
            f"Feature contract {name} 'features' must be a list, got {type(contract['features'])}"
        )

    if len(contract["features"]) != contract["n_features"]:
        raise FeatureContractError(
            f"Feature contract {name} mismatch: features list has {len(contract['features'])} items "
            f"but n_features={contract['n_features']}"
        )

    return contract


def get_feature_list(name: str = "ml_v1") -> List[str]:
    """
    Get ordered list of feature names from contract.

    Args:
        name: Contract name

    Returns:
        Ordered list of feature names

    Example:
        >>> features = get_feature_list("ml_v1")
        >>> len(features)
        28
    """
    contract = load_feature_contract(name)
    return contract["features"]


def validate_contract_compliance(
    feature_names: List[str],
    contract_name: str = "ml_v1",
    strict_order: bool = True
) -> Dict[str, Any]:
    """
    Validate that provided feature names comply with contract.

    Args:
        feature_names: List of feature names to validate
        contract_name: Name of contract to validate against
        strict_order: If True, require exact order match

    Returns:
        Dict with validation results:
            - compliant: bool
            - missing: List[str] (features in contract but not in input)
            - extra: List[str] (features in input but not in contract)
            - order_mismatch: bool (if strict_order=True)
    """
    contract = load_feature_contract(contract_name)
    contract_features = contract["features"]

    provided_set = set(feature_names)
    contract_set = set(contract_features)

    missing = list(contract_set - provided_set)
    extra = list(provided_set - contract_set)

    order_mismatch = False
    if strict_order and contract.get("enforce_order", True):
        order_mismatch = feature_names != contract_features

    compliant = (len(missing) == 0 and len(extra) == 0 and not order_mismatch)

    return {
        "compliant": compliant,
        "missing": missing,
        "extra": extra,
        "order_mismatch": order_mismatch,
        "expected_count": len(contract_features),
        "provided_count": len(feature_names)
    }
