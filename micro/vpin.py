
import pandas as pd
import numpy as np

def compute_vpin(trades: pd.DataFrame, bucket_size: int = 50) -> float:
    """
    trades: DataFrame with columns ['timestamp','size','side' (1 buy/-1 sell)]
    returns 0..1. On error -> return 0.0
    
    VPIN = (1/(N*V)) * sum(|V_buy_bucket - V_sell_bucket|)
    """
    try:
        if trades.empty or 'size' not in trades.columns or 'side' not in trades.columns:
            return 0.0
            
        # Ensure sorted
        # trades = trades.sort_values('timestamp') # Assuming stream is sorted for O(N)
        
        # Calculate Volume Buckets
        total_vol = trades['size'].sum()
        if total_vol == 0: return 0.0
        
        # For simplicity in "Tiny API", we just bin by N trades if volume not available, 
        # or we implement strict volume bucketing.
        # Let's do Strict Volume Bucketing (O(N))
        
        # Current bucket state
        current_vol = 0
        buy_vol = 0
        sell_vol = 0
        bucket_imbalances = []
        
        # Vectorized accumulation might be hard with volume-triggers, so we loop or use cumsum
        # O(N) approach: CumSum -> Modulo
        
        trades['signed_vol'] = trades['size'] * trades['side']
        trades['cum_vol'] = trades['size'].cumsum()
        
        # Determine bucket target V
        # If we want 50 buckets per day, V = Total / 50. Or fixed size?
        # Prompt says "bucket size V_bucket", passed as 50? Assume N=50 buckets implies V = Total/50
        n_buckets = bucket_size
        v_bucket = total_vol / n_buckets
        
        if v_bucket == 0: return 0.0
        
        # Assign bucket IDs
        trades['bucket_id'] = (trades['cum_vol'] // v_bucket).astype(int)
        
        # GroupBy Bucket
        grouped = trades.groupby('bucket_id')[['size', 'signed_vol']].sum()
        
        # Reconstruct Buy/Sell per bucket
        # signed_vol = Buy - Sell
        # size = Buy + Sell
        # => Buy = (size + signed) / 2
        # => Sell = (size - signed) / 2
        
        grouped['buy_vol'] = (grouped['size'] + grouped['signed_vol']) / 2
        grouped['sell_vol'] = (grouped['size'] - grouped['signed_vol']) / 2
        
        grouped['imbalance'] = (grouped['buy_vol'] - grouped['sell_vol']).abs()
        
        sum_imbalance = grouped['imbalance'].sum()
        
        # VPIN = Sum(|OI|) / (N * V) = Sum(|OI|) / TotalVolume
        vpin = sum_imbalance / total_vol
        
        return float(np.clip(vpin, 0.0, 1.0))
        
    except Exception:
        return 0.0
