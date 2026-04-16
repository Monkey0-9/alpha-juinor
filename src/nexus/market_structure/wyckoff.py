
import pandas as pd
from typing import Dict

def structure_filter(df: pd.DataFrame) -> Dict[str, bool]:
    """Return {'allow_long': bool, 'allow_short': bool}"""
    try:
        # PANDAS HYGIENE
        if df.empty or 'Close' not in df or 'Volume' not in df:
            return {'allow_long': True, 'allow_short': True}
            
        close = df['Close']
        vol = df['Volume']
        if len(close) < 20: return {'allow_long': True, 'allow_short': True}
        
        # Wyckoff Logic: Volume/Price Divergence
        # 1. Price Highs with low Volume -> Weak demand (Block Long)
        recent_high = close.rolling(20).max().iloc[-1]
        curr_price = close.iloc[-1]
        curr_vol = vol.iloc[-1]
        avg_vol = vol.rolling(20).mean().iloc[-1]
        
        at_high = curr_price >= recent_high * 0.99
        low_vol = curr_vol < avg_vol * 0.8
        
        allow_long = not (at_high and low_vol)
        
        # 2. Price Lows with low Volume -> Weak supply (Block Short)
        recent_low = close.rolling(20).min().iloc[-1]
        at_low = curr_price <= recent_low * 1.01
        
        allow_short = not (at_low and low_vol)
        
        return {'allow_long': allow_long, 'allow_short': allow_short}
        
    except Exception:
        return {'allow_long': True, 'allow_short': True}
