"""
Feature validation schema and utilities for institutional trading pipeline.
Ensures all required ML features are present before model execution.
"""

from typing import List, Set, Tuple

import pandas as pd

# Required institutional features for all trading models
REQUIRED_FEATURES = {
    # Returns
    "returns_1d",
    "returns_5d",
    "returns_20d",
    "returns_60d",
    # Volatility
    "volatility_20d",
    "volatility_60d",
    # Market regime
    "price_to_sma_200",
    "rsi_14",
    # Liquidity
    "volume_ratio_20d",
    "bid_ask_spread_bps",
    # Momentum
    "momentum_20d",
    "momentum_60d",
    # Mean reversion
    "zscore_20d",
    "zscore_60d",
    # Data quality
    "data_quality_score",
}

# Optional features that may be available
OPTIONAL_FEATURES = {
    "ml_probability",
    "ensemble_prediction",
    "option_flow_signal",
    "insider_transaction_signal",
    "order_flow_imbalance",
    "smart_money_signal",
    "crisis_signal",
}


def validate_feature_schema(
    df: pd.DataFrame,
    required_features: Set[str] = None,
    optional_features: Set[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame contains all required features.

    Args:
        df: DataFrame to validate
        required_features: Set of required feature names (defaults to REQUIRED_FEATURES)
        optional_features: Set of optional feature names (defaults to OPTIONAL_FEATURES)

    Returns:
        Tuple of (is_valid, missing_features)
        - is_valid: True if all required features present, False otherwise
        - missing_features: List of missing required features
    """
    if required_features is None:
        required_features = REQUIRED_FEATURES
    if optional_features is None:
        optional_features = OPTIONAL_FEATURES

    df_columns = set(df.columns)
    missing_required = list(required_features - df_columns)

    is_valid = len(missing_required) == 0

    return is_valid, missing_required


def get_available_optional_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of optional features that are available in the DataFrame.

    Args:
        df: DataFrame to check

    Returns:
        List of available optional features
    """
    df_columns = set(df.columns)
    return list(OPTIONAL_FEATURES & df_columns)


def validate_feature_dtypes(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that features have appropriate data types (numeric).

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, invalid_features)
    """
    invalid_features = []

    for col in REQUIRED_FEATURES:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                invalid_features.append(col)

    return len(invalid_features) == 0, invalid_features


def validate_no_missing_values(
    df: pd.DataFrame, allow_optional_missing: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that required features have no missing values.

    Args:
        df: DataFrame to validate
        allow_optional_missing: If True, optional features can have missing values

    Returns:
        Tuple of (is_valid, features_with_missing)
    """
    features_with_missing = []

    check_features = REQUIRED_FEATURES
    if not allow_optional_missing:
        check_features = REQUIRED_FEATURES | OPTIONAL_FEATURES

    for col in check_features:
        if col in df.columns and df[col].isna().any():
            features_with_missing.append(col)

    return len(features_with_missing) == 0, features_with_missing


def validate_feature_ranges(df: pd.DataFrame) -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Validate that features are within reasonable ranges (detect data corruption).

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, invalid_ranges) where invalid_ranges is list of (feature, reason)
    """
    invalid_ranges = []

    # Volatility should be reasonable (0.01 to 5.0)
    for col in ["volatility_20d", "volatility_60d"]:
        if col in df.columns:
            if (df[col] < 0).any() or (df[col] > 5.0).any():
                invalid_ranges.append((col, "Volatility out of range [0, 5.0]"))

    # RSI should be 0-100
    if "rsi_14" in df.columns:
        if (df["rsi_14"] < 0).any() or (df["rsi_14"] > 100).any():
            invalid_ranges.append(("rsi_14", "RSI out of range [0, 100]"))

    # Z-score should generally be -5 to 5 (but allow outliers)
    for col in ["zscore_20d", "zscore_60d"]:
        if col in df.columns:
            if (df[col].abs() > 20).any():
                invalid_ranges.append((col, "Z-score extreme (>20)"))

    return len(invalid_ranges) == 0, invalid_ranges


def full_validation(df: pd.DataFrame) -> Tuple[bool, dict]:
    """
    Run complete feature validation suite.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, validation_report)
    """
    report = {}

    # Schema validation
    schema_valid, missing = validate_feature_schema(df)
    report["schema"] = {"valid": schema_valid, "missing_features": missing}

    # Data type validation
    dtype_valid, invalid_dtypes = validate_feature_dtypes(df)
    report["dtypes"] = {"valid": dtype_valid, "invalid_features": invalid_dtypes}

    # Missing values validation
    missing_valid, cols_with_missing = validate_no_missing_values(df)
    report["missing_values"] = {"valid": missing_valid, "features": cols_with_missing}

    # Range validation
    range_valid, invalid_ranges = validate_feature_ranges(df)
    report["ranges"] = {"valid": range_valid, "issues": invalid_ranges}

    # Optional features
    report["optional_features"] = get_available_optional_features(df)

    # Overall result
    is_valid = all(
        r["valid"]
        for r in [
            report["schema"],
            report["dtypes"],
            report["missing_values"],
            report["ranges"],
        ]
    )

    return is_valid, report
