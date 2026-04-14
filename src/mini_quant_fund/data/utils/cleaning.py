"""
Institutional Data Cleaning Utilities.

Provides robust data sanitization functions to prevent NaN/inf propagation
through the trading pipeline. Used at every data transformation stage.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def institutional_clean(series: Union[pd.Series, pd.DataFrame], name: str = "unknown") -> Union[pd.Series, pd.DataFrame]:
    """
    Institutional-grade data cleaning with context-aware NaN/inf handling.

    Args:
        series: Input data to clean
        name: Context name for logging (e.g., "returns", "prices")

    Returns:
        Cleaned series/dataframe with NaN/inf handled appropriately

    Strategy:
    - Type safety: Convert to float
    - Empty check: Return zero-indexed series if empty
    - Infinity handling: Replace inf/-inf with NaN
    - NaN strategy: Context-dependent (returns drop NaNs, prices fill)
    """
    # Type safety
    if isinstance(series, pd.DataFrame):
        clean = series.astype(float)
    else:
        clean = series.astype(float)

    # Empty check
    if clean.empty:
        if isinstance(clean, pd.DataFrame):
            return pd.DataFrame(0.0, index=series.index, columns=series.columns)
        else:
            return pd.Series(0.0, index=series.index)

    # Infinity handling
    clean = clean.replace([np.inf, -np.inf], np.nan)

    # NaN strategy (context-dependent)
    if "return" in name.lower():
        # Returns: drop NaNs (missing returns are invalid)
        if isinstance(clean, pd.DataFrame):
            clean = clean.dropna()
        else:
            clean = clean.dropna()
    else:
        # Prices/Volumes: fill forward then backward with zeros
        clean = clean.ffill().fillna(0.0)

    return clean


def safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Safe percentage change with explicit fill_method=None and inf/nan removal.

    Args:
        series: Price series
        periods: Periods for pct_change

    Returns:
        Clean returns series
    """
    returns = series.pct_change(periods=periods, fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns.astype(float)


def validate_data_quality(data: Union[pd.Series, pd.DataFrame], stage: str) -> bool:
    """
    Validate data quality at pipeline stage.

    Args:
        data: Data to validate
        stage: Pipeline stage name

    Returns:
        True if data passes validation, False otherwise
    """
    if data is None:
        print(f"VALIDATION FAILED: {stage} - Data is None")
        return False

    if isinstance(data, pd.Series) and data.empty:
        print(f"VALIDATION FAILED: {stage} - Series is empty")
        return False

    if isinstance(data, pd.DataFrame) and data.empty:
        print(f"VALIDATION FAILED: {stage} - DataFrame is empty")
        return False

    # Check for NaN/inf
    if isinstance(data, pd.Series):
        if data.isna().any():
            print(f"VALIDATION WARNING: {stage} - Contains NaN values")
        if np.isinf(data.values).any():
            print(f"VALIDATION FAILED: {stage} - Contains infinite values")
            return False
    else:
        if data.isna().any().any():
            print(f"VALIDATION WARNING: {stage} - Contains NaN values")
        if np.isinf(data.values).any():
            print(f"VALIDATION FAILED: {stage} - Contains infinite values")
            return False

    return True
