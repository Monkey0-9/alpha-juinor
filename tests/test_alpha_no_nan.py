
import pytest
import pandas as pd
import numpy as np
from strategies.alpha import TrendAlpha, MeanReversionAlpha, RSIAlpha

@pytest.mark.parametrize("alpha_class", [TrendAlpha, MeanReversionAlpha, RSIAlpha])
def test_alpha_no_nan_output(alpha_class):
    prices = pd.Series(np.random.randn(300).cumsum() + 100)
    prices.iloc[10:20] = np.nan
    prices.iloc[50] = np.inf
    
    alpha = alpha_class()
    if isinstance(alpha, (TrendAlpha, MeanReversionAlpha)):
        # MeanReversionAlpha uses windows, give it enough data
        pass
        
    res = alpha.compute(prices)
    assert not res.isna().any()
    assert not np.isinf(res).any()
    assert res.dtype == float
    assert res.min() >= 0.0
    assert res.max() <= 1.0

def test_alpha_empty_input_aligned():
    prices = pd.Series([], dtype=float)
    alpha = RSIAlpha()
    res = alpha.compute(prices)
    assert res.empty
    # Even if empty, it should return a series aligned to input if possible
    # but here input is empty, so output is empty.
