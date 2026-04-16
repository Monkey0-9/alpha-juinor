
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("TIMEUTILS")

def safe_infer_freq(df: pd.DataFrame, target_freq: str = 'B') -> pd.DataFrame:
    """
    Ensures DataFrame has a DatetimeIndex with correct timezone (UTC)
    and specified frequency (default 'B' for business days).

    Fills missing steps with forward-fill.
    """
    if df is None or df.empty:
        return df

    # 1. Ensure Index is Datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert index to DatetimeIndex: {e}")
            return df

    # 2. Normalize Timezone (UTC)
    if df.index.tz is None:
        # Assume UTC if naive, or localize carefully
        # Institutional standard: Always UTC
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # 3. Sort
    df = df.sort_index()

    # 4. Remove Duplicates
    if df.index.duplicated().any():
        logger.warning(f"Duplicate index found in safe_infer_freq. Keeping last.")
        df = df[~df.index.duplicated(keep='last')]

    # 5. Resample/Asfreq to impose frequency
    try:
        if df.index.freq != target_freq:
            # Reindex to full business day range to fill gaps
            full_idx = pd.date_range(start=df.index[0], end=df.index[-1], freq=target_freq, tz='UTC')

            # Reindex, forward filling values to handle missing days (stubs)
            df = df.reindex(full_idx, method='ffill')

            # Explicitly set freq (reindex usually sets it, but be safe)
            df.index.freq = target_freq

    except Exception as e:
        logger.error(f"Failed to set frequency {target_freq}: {e}")

    return df

def ensure_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper for strict business day enforcement."""
    return safe_infer_freq(df, target_freq='B')
