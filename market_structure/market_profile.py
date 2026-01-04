
import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_market_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return {'inside_value_area': bool, 'poc': float, 'vah': float, 'val': float}
    """
    try:
        if df.empty or 'Close' not in df or 'Volume' not in df:
            return {'inside_value_area': False}
            
        # Simplified Volume Profile on last N bars
        df_sub = df.tail(50)
        prices = df_sub['Close']
        volumes = df_sub['Volume']
        
        hist, bin_edges = np.histogram(prices, bins=15, weights=volumes)
        poc_idx = np.argmax(hist)
        poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
        
        # Value Area (70%)
        total_vol = hist.sum()
        target = 0.7 * total_vol
        
        curr_vol = hist[poc_idx]
        left, right = poc_idx, poc_idx
        
        while curr_vol < target:
            l_vol = hist[left-1] if left > 0 else 0
            r_vol = hist[right+1] if right < len(hist)-1 else 0
            
            if l_vol > r_vol and left > 0:
                left -= 1
                curr_vol += l_vol
            elif right < len(hist)-1:
                right += 1
                curr_vol += r_vol
            else:
                break
                
        val = bin_edges[left]
        vah = bin_edges[right+1]
        
        last_price = prices.iloc[-1]
        inside = val <= last_price <= vah
        
        return {
            'inside_value_area': bool(inside),
            'poc': float(poc_price),
            'vah': float(vah),
            'val': float(val)
        }
        
    except Exception:
         return {'inside_value_area': False}
