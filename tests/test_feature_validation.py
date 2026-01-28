import pytest
import pandas as pd
import numpy as np
from data.processors.validator import validate_features, sanitize_numeric_df, ensure_adjusted_close, FeatureValidationError

def test_sanitize_numeric_df():
    df = pd.DataFrame({
        'A': [1.0, np.inf, -np.inf, np.nan],
        'B': ['1', '2', 'invalid', '4'],
        'C': [np.nan, np.nan, np.nan, np.nan]
    })

    sanitized = sanitize_numeric_df(df)

    # Check that inf/nan are handled
    assert np.isnan(sanitized.loc[1, 'A'])

    # Check that column B is coerced to numeric
    assert sanitized['B'].dtype == float or sanitized['B'].dtype == np.float64
    assert sanitized.loc[3, 'B'] == 4.0

    # In the example, row 2 becomes all nan after inf replacement and coercion
    assert 2 not in sanitized.index
    assert 3 in sanitized.index

def test_ensure_adjusted_close():
    # Case 1: Adj Close exists
    df1 = pd.DataFrame({'Close': [100.0], 'Adj Close': [98.0]})
    res1 = ensure_adjusted_close(df1)
    assert 'adjusted_close' in res1.columns
    assert res1['adjusted_close'].iloc[0] == 98.0
    assert not res1['__adjusted_close_fallback__'].iloc[0]

    # Case 2: Adj Close missing
    df2 = pd.DataFrame({'Close': [100.0]})
    res2 = ensure_adjusted_close(df2)
    assert 'adjusted_close' in res2.columns
    assert res2['adjusted_close'].iloc[0] == 100.0
    assert res2['__adjusted_close_fallback__'].iloc[0]

    # Case 3: Both missing
    with pytest.raises(FeatureValidationError):
        ensure_adjusted_close(pd.DataFrame({'Volume': [1000]}))

def test_inf_and_missing_adjusted_close():
    # Combined check
    df = pd.DataFrame({
        'Close': [100.0, np.inf, 200.0],
        'volume': [1000, 2000, 3000]
    })

    validated = validate_features(df, required_features=['adjusted_close', 'volume'])

    assert 'adjusted_close' in validated.columns
    assert validated.loc[1, 'adjusted_close'] == 0.0 # inf becomes nan becomes fillna(0)
    assert validated['adjusted_close'].dtype == np.float32

def test_missing_features_raises():
    df = pd.DataFrame({'A': [1, 2]})
    with pytest.raises(FeatureValidationError, match="Missing required features"):
        validate_features(df, required_features=['B'])
