# data/validator.py
import pandas as pd
import numpy as np
from typing import List, Dict

class DataValidator:
    """
    Validate and clean market data.
    """
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, ticker: str = "UNKNOWN") -> Dict[str, any]:
        """
        Validate OHLCV DataFrame.
        Returns dict with 'valid' (bool) and 'issues' (list).
        """
        issues = []
        
        # 1. Check required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return {"valid": False, "issues": issues, "ticker": ticker}
        
        # 2. Check for NaNs
        nan_counts = df[required].isnull().sum()
        if nan_counts.any():
            issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
        
        # 3. Check for negative prices
        for col in ["Open", "High", "Low", "Close"]:
            if (df[col] < 0).any():
                issues.append(f"Negative {col} prices detected")
        
        # 4. Check OHLC relationship
        if not df.empty:
            invalid_bars = (df["High"] < df["Low"]) | (df["High"] < df["Close"]) | (df["Low"] > df["Close"])
            if invalid_bars.any():
                issues.append(f"Invalid OHLC relationships in {invalid_bars.sum()} bars")
        
        # 5. Check for extreme volume spikes (>100x median)
        if not df.empty and len(df) > 20:
            median_vol = df["Volume"].median()
            if median_vol > 0:
                extreme_vol = (df["Volume"] > median_vol * 100).sum()
                if extreme_vol > 0:
                    issues.append(f"Extreme volume spikes: {extreme_vol} bars")
        
        valid = len(issues) == 0
        return {"valid": valid, "issues": issues, "ticker": ticker}
    
    @staticmethod
    def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data:
        - Forward fill NaNs
        - Clip negative prices to previous close
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Forward fill NaNs
        df = df.fillna(method='ffill')
        
        # Clip negative prices
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0.01)  # Minimum price 1 cent
        
        return df
