import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from numba import njit

@njit
def fast_ts_mean(x, d):
    res = np.empty_like(x)
    for i in range(len(x)):
        if i < d - 1:
            res[i] = np.nan
        else:
            res[i] = np.mean(x[i-d+1:i+1])
    return res

@njit
def fast_ts_std(x, d):
    res = np.empty_like(x)
    for i in range(len(x)):
        if i < d - 1:
            res[i] = np.nan
        else:
            res[i] = np.std(x[i-d+1:i+1])
    return res

class AlphaDSL:
    """Alpha Expression Language (WorldQuant Style) - Numba Accelerated"""

    def __init__(self, data: pd.DataFrame):
        self.data = data # Expected columns: open, high, low, close, volume, returns
        self.raw_data = {c: data[c].values for c in data.columns}
        self.local_dict = {
            "ts_mean": self.ts_mean,
            "ts_std": self.ts_std,
            "ts_delta": self.ts_delta,
            "rank": self.rank,
            "zscore": self.zscore,
            "close": self.raw_data["close"],
            "open": self.raw_data["open"],
            "high": self.raw_data["high"],
            "low": self.raw_data["low"],
            "volume": self.raw_data["volume"]
        }

    def ts_mean(self, x, d: int):
        if isinstance(x, pd.Series): x = x.values
        return fast_ts_mean(x, d)

    def ts_std(self, x, d: int):
        if isinstance(x, pd.Series): x = x.values
        return fast_ts_std(x, d)

    def ts_delta(self, x, d: int):
        if isinstance(x, pd.Series): x = x.values
        res = np.empty_like(x)
        res[:d] = np.nan
        res[d:] = x[d:] - x[:-d]
        return res

    def rank(self, x):
        if isinstance(x, pd.Series): x = x.values
        return pd.Series(x).rank(pct=True).values

    def zscore(self, x):
        if isinstance(x, pd.Series): x = x.values
        return (x - np.nanmean(x)) / np.nanstd(x)

    def evaluate(self, expression: str, return_series: bool = True) -> Union[pd.Series, np.ndarray]:
        """
        Evaluate a string expression.
        """
        res = eval(expression, {"__builtins__": None, "np": np}, self.local_dict)
        if return_series:
            return pd.Series(res, index=self.data.index)
        return res
