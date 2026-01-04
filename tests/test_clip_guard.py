
import pytest
import pandas as pd
import numpy as np
from strategies.alpha import safe_clip

def test_clip_guard_all_nan():
    index = pd.date_range("2023-01-01", periods=10)
    raw = pd.Series([np.nan] * 10, index=index)
    res = safe_clip(raw, index)
    assert len(res) == 10
    assert (res == 0.0).all()

def test_clip_guard_partial_nan():
    index = pd.date_range("2023-01-01", periods=3)
    raw = pd.Series([0.5, np.nan, 0.8], index=index)
    # Note: safe_clip drops NaNs before clipping
    # So the result will be empty if we didn't have the guard.
    # Actually, current safe_clip:
    # raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
    # if raw.empty: return aligned_zeros
    # return raw.clip(0, 1)
    
    res = safe_clip(raw, index)
    # If 0.5 and 0.8 remain, it's NOT empty.
    # But wait, raw.clip(0, 1) will only have 2 elements.
    # The requirement says "return a zero Series aligned to the expected index" ONLY if it results in an EMPTY series.
    # Otherwise it returns the clipped series. 
    # Usually we want aligned series for the whole index.
    # Let's check the requirement again.
    # "If this results in an empty Series, return a zero Series aligned to the expected index"
    
    assert len(res) == 2 # 0.5 and 0.8 remain
    assert res.iloc[0] == 0.5
    assert res.iloc[1] == 0.8
