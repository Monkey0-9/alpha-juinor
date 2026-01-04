# utils/time.py
import pandas as pd
from datetime import datetime

def to_utc(ts):
    """
    Standardizes any timestamp to a UTC-aware pd.Timestamp.
    Institutional Invariant: ALL internal comparisons must be UTC.
    """
    if ts is None:
        return None
        
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def get_now_utc():
    """Returns the current time shifted to UTC."""
    return pd.Timestamp.now(tz="UTC")
