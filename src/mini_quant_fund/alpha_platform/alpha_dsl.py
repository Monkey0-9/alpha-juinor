import numpy as np
import pandas as pd
from typing import Dict, Any

class AlphaDSL:
    """Alpha Expression Language (WorldQuant Style)"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data # Expected columns: open, high, low, close, volume, returns
        
    def ts_mean(self, x: pd.Series, d: int) -> pd.Series:
        return x.rolling(window=d).mean()
        
    def ts_std(self, x: pd.Series, d: int) -> pd.Series:
        return x.rolling(window=d).std()
        
    def rank(self, x: pd.Series) -> pd.Series:
        return x.rank(pct=True)
        
    def zscore(self, x: pd.Series) -> pd.Series:
        return (x - x.mean()) / x.std()
        
    def group_neutralize(self, x: pd.Series, group: pd.Series) -> pd.Series:
        return x - x.groupby(group).transform("mean")
        
    def winsorize(self, x: pd.Series, pct: float) -> pd.Series:
        lower = x.quantile(pct)
        upper = x.quantile(1 - pct)
        return x.clip(lower, upper)

    def evaluate(self, expression: str) -> pd.Series:
        """
        Evaluate a string expression.
        Simplified: this would normally be a real parser.
        For now, we assume simple Python-like expressions using local methods.
        """
        # This is a dangerous implementation, but serves as a scaffold
        # In production, use a safe AST-based parser
        local_dict = {
            "ts_mean": self.ts_mean,
            "ts_std": self.ts_std,
            "rank": self.rank,
            "zscore": self.zscore,
            "group_neutralize": self.group_neutralize,
            "winsorize": self.winsorize,
            "close": self.data["close"],
            "open": self.data["open"],
            "high": self.data["high"],
            "low": self.data["low"],
            "volume": self.data["volume"]
        }
        return eval(expression, {"__builtins__": None}, local_dict)
