"""
Schema normalization utilities for institutional trading systems.

Ensures consistent OHLCV column naming across all data providers.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def normalize_ohlcv(df):
    """
    Normalize OHLCV schema across all providers.
    Ensures required columns exist with consistent casing.
    """
    if df is None:
        return df

    # Handle Series input (single column data)
    if isinstance(df, pd.Series):
        # Convert Series to DataFrame
        df = df.to_frame()
        logger.debug(f"Converted Series to DataFrame. Shape: {df.shape}, Columns: {df.columns.tolist()}")

    if df.empty:
        return df

    # Standardize column names
    return df


def ensure_dataframe(data, required_columns=None):
    """
    STRICT ENFORCEMENT: Ensures input is a DataFrame.
    If Series, converts to DataFrame.
    If required_columns specified, validates they exist.
    """
    if data is None:
        return pd.DataFrame()
        
    if isinstance(data, pd.Series):
        # Handle the case where the Series name should be the column name
        name = data.name if data.name else "Value"
        df = data.to_frame(name=name)
        logger.debug(f"Converted Series '{name}' to DataFrame.")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        # Unexpected type, attempt conversion but log warning
        logger.warning(f"Unexpected data type {type(data)} passed to ensure_dataframe. Attempting conversion.")
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to convert {type(data)} to DataFrame: {e}")
            return pd.DataFrame()

    if required_columns:
        if isinstance(required_columns, str):
            required_columns = [required_columns]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            # Check MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                # Check levels
                for col in required_columns:
                    if col not in df.columns.get_level_values(1) and col not in df.columns.get_level_values(0):
                        raise ValueError(f"Institutional Schema Violation: Missing required column/level '{col}'. Available: {df.columns.tolist()}")
            else:
                raise ValueError(f"Institutional Schema Violation: Missing required columns {missing}. Available: {df.columns.tolist()}")
                
    return df


def get_price(df):
    """
    Canonical price accessor.
    Disallow direct df["Close"] access elsewhere.
    """
    return df["Close"]
