import pandas as pd
import numpy as np
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FeatureValidationError(Exception):
    pass

def sanitize_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace ±inf with NaN, coerce objects to numeric where possible, drop all-NaN rows.
    """
    if df.empty:
        return df

    # Coerce to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Replace ±inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are completely NaN
    df = df.dropna(how='all')

    return df

def ensure_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure adjusted_close exists. If missing, fill from close and set fallback flag.
    """
    df = df.copy()

    # Handle both capitalized and lowercase
    close_col = 'Close' if 'Close' in df.columns else ('close' if 'close' in df.columns else None)
    adj_close_col = 'Adj Close' if 'Adj Close' in df.columns else ('adjusted_close' if 'adjusted_close' in df.columns else None)

    if not adj_close_col:
        if not close_col:
            raise FeatureValidationError("Neither 'Close' nor 'Adj Close' columns found in data.")

        # Use close as fallback
        df['adjusted_close'] = df[close_col]
        df['__adjusted_close_fallback__'] = True
        logger.debug("Adjusted close missing, using raw close as fallback.")
    else:
        # Standardize name
        if adj_close_col != 'adjusted_close':
            df['adjusted_close'] = df[adj_close_col]

        df['__adjusted_close_fallback__'] = False

    return df

def validate_features(X: pd.DataFrame, required_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Ensure required feature set exists, enforce deterministic column ordering and cast to float32.
    """
    if X.empty:
         raise FeatureValidationError("Input features X is empty")

    X = X.copy()

    # Sanitize
    X = sanitize_numeric_df(X)

    # Ensure adjusted close (legacy requirement for some models)
    try:
        X = ensure_adjusted_close(X)
    except FeatureValidationError:
        # If not a price DF, skip adjusted close check
        pass

    # Handle NaNs (institutional default: fill 0.0 for robustness)
    if X.isnull().any().any():
        X.fillna(0.0, inplace=True)

    # Check for required columns
    if required_features:
        missing = [c for c in required_features if c not in X.columns]
        if missing:
            raise FeatureValidationError(f"Missing required features: {missing}")
        # Enforce order
        X = X[required_features]

    # Enforce float32
    X = X.astype(np.float32)

    return X
