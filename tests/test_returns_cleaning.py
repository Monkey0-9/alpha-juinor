
import pytest
import pandas as pd
import numpy as np
from strategies.alpha import safe_pct_change, safe_clip

def test_safe_pct_change_no_inf():
    s = pd.Series([100, 0, 100, 110])
    # 0 -> 100 would be inf without fill_method=None handling
    res = safe_pct_change(s)
    assert not np.isinf(res).any()
    assert not res.isna().any()

def test_safe_pct_change_empty():
    s = pd.Series([], dtype=float)
    res = safe_pct_change(s)
    assert res.empty
    assert res.dtype == float

def test_safe_clip_empty_returns_zeros():
    index = pd.date_range("2023-01-01", periods=5)
    raw = pd.Series([np.nan, np.nan], index=[index[0], index[1]])
    res = safe_clip(raw, index)
    assert len(res) == len(index)
    assert (res == 0.0).all()
    assert res.index.equals(index)

def test_safe_clip_bounds():
    index = pd.date_range("2023-01-01", periods=3)
    raw = pd.Series([1.5, -0.5, 0.5], index=index)
    res = safe_clip(raw, index)
    assert res.max() <= 1.0
    assert res.min() >= 0.0
    assert res.iloc[0] == 1.0
    assert res.iloc[1] == 0.0
    assert res.iloc[2] == 0.5
