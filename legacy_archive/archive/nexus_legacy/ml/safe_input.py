"""
ml/safe_input.py

Feature alignment and safety adapter for ML models.
Ensures that input features match model expectations by:
- Reordering columns
- Filling missing features with imputations
- Handling extra features
- Logging mismatches
"""

import numpy as np
import pandas as pd
import logging
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Dict, Any

logger = logging.getLogger("ML_SAFE_INPUT")

def align_features(X: pd.DataFrame, expected: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Align input features to the expected list.

    Args:
        X: current pipeline output (rows x cols)
        expected: list of feature names the model expects in order

    Returns:
        X_aligned: Imputed and reordered DataFrame
        missing: List of missing feature names
        extra: List of extra feature names
    """
    X = X.copy()

    # If X is empty or all NaN, return with expected schema
    if X.empty:
        X_aligned = pd.DataFrame(np.nan, index=[0], columns=expected)
    else:
        provided_features = list(X.columns)
        missing = [f for f in expected if f not in provided_features]
        extra = [c for c in provided_features if c not in expected]

        # Add missing with NaN
        for f in missing:
            X[f] = np.nan

        # Reindex to expected order (drop extras)
        X_aligned = X.reindex(columns=expected)

    # Impute: use median (per user request) or 0 as ultimate fallback
    try:
        # Check if we have anything to impute
        if X_aligned.isnull().any().any():
            # If multi-row, we can use median. If single-row, we might need a better policy.
            # SimpleImputer with median fits on the data provided.
            # If all are NaN, median will stay NaN. Use constant 0 as final guard.

            # 1. Try median
            X_imputed = X_aligned.fillna(X_aligned.median())

            # 2. Final guard: fill remaining NaNs with 0.0
            X_imputed = X_imputed.fillna(0.0)

            return X_imputed, missing, extra
        else:
            return X_aligned, [], []
    except Exception as e:
        logger.error(f"Feature alignment/imputation failed: {e}")
        # Final fallback: fillna(0)
        return X_aligned.fillna(0.0), missing, extra


def distributional_sanity_check(mu: float, sigma: float, cvar: float) -> Tuple[bool, str]:
    """
    Validate that predicted distribution parameters are within reasonable bounds.
    """
    # Reasonable bounds for daily parameters
    # mu: +/- 20%
    if abs(mu) > 0.20:
        return False, f"mu {mu} out of bounds"

    # sigma: 0 to 500% (annualized) -> roughly 0 to 30% daily
    if sigma <= 0 or sigma > 0.30:
        return False, f"sigma {sigma} out of bounds"

    # cvar: negative but not more than -100%
    if cvar > 0 or cvar < -1.0:
        return False, f"cvar {cvar} out of bounds"

    return True, "OK"
