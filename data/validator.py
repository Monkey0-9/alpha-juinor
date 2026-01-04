
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict

logger = logging.getLogger("data_validator")

class DataValidator:
    """
    Utility for fail-fast validation of market data.
    Ensures OHLC consistency and numeric integrity.
    """
    
    @staticmethod
    def validate_ohlc(df: pd.DataFrame, ticker: str = "Unknown") -> pd.DataFrame:
        """
        Validates OHLC data:
        1. High >= Low
        2. Open/Close within [Low, High] range
        3. Identifies and removes malformed rows.
        """
        if df.empty:
            return df
            
        initial_count = len(df)
        
        # 1. Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Check High >= Low
        mask = df['High'] >= df['Low']
        
        # 3. Check Open/Close within range
        mask &= (df['Open'] >= df['Low']) & (df['Open'] <= df['High'])
        mask &= (df['Close'] >= df['Low']) & (df['Close'] <= df['High'])
        
        # 4. Filter out NaNs if any after coercion
        mask &= df[['Open', 'High', 'Low', 'Close']].notna().all(axis=1)
        
        cleaned_df = df[mask].copy()
        dropped_count = initial_count - len(cleaned_df)
        
        if dropped_count > 0:
            logger.warning(
                f"[VALIDATOR] Dropped {dropped_count} malformed rows for {ticker}. "
                f"Reason: OHLC inconsistency or NaNs."
            )
            
        return cleaned_df

    @staticmethod
    def sanitize_series(s: pd.Series, ticker: str = "Unknown") -> pd.Series:
        """
        Standard sanitation for numeric series in the pipeline.
        - Casts to float
        - Replaces inf with NaN
        - Drops NaN
        """
        initial_val = len(s)
        s = pd.to_numeric(s, errors='coerce').astype(float, copy=False)
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        
        dropped = initial_val - len(s)
        if dropped > 0:
            logger.debug(f"[VALIDATOR] {ticker}: {dropped} bars dropped during sanitation.")
            
        return s
