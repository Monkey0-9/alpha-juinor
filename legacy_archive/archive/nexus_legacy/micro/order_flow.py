
import pandas as pd
import numpy as np

def compute_hawkes_intensity(timestamps: pd.Series, decay: float = 1.0) -> float:
    """
    Hawkes intensity (research stub).
    λ(t) = μ + Σ_{t_i < t} α e^{-β(t − t_i)}
    Returns current intensity scalar.
    """
    try:
        # Simplified recursive calculation for O(N) instead of O(N^2)
        # A(t_k) = e^{-β(t_k - t_{k-1})} * A(t_{k-1}) + α
        if timestamps.empty: return 0.0
        
        # Convert to seconds/deltas
        times = pd.to_numeric(timestamps) / 1e9 # ns to s
        if len(times) < 2: return 0.0
        
        deltas = times.diff().dropna().values
        
        intensity = 0.0
        alpha = 0.5
        beta = decay
        
        # Only compute last point intensity for speed check?
        # If we need history, we scan. For 'stub', we do concise loop.
        current_excitation = 0.0
        for dt in deltas:
            current_excitation = np.exp(-beta * dt) * (current_excitation + alpha)
            
        mu = 0.1 # Baseline
        return mu + current_excitation
    except Exception:
        return 0.0

def measure_adverse_selection(trades: pd.DataFrame, quotes: pd.DataFrame, horizon_sec: int = 5) -> float:
    """
    Price change against trade direction after H seconds.
    """
    try:
        if trades.empty or quotes.empty: return 0.0
        # Placeholder O(1) stub
        return 0.0
    except Exception:
        return 0.0
