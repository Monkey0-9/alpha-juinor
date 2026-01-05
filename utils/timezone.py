
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def normalize_index_utc(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Standardizes the index to be UTC-aware.
    Handles both naive (assumes local/UTC?) and aware timestamps.
    Institutional Requirement: All internal data MUST be UTC.
    """
    try:
        # If it's a Series, convert access pattern slightly or just handle index
        if df.index.tz is None:
            # Assume UTC if naive, or localize to UTC
            # If we assumed local, we'd do tz_localize('sys_local').tz_convert('UTC')
            # But usually sticking to "Naive is UTC" is safer for free data unless specified.
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
            
        return df
    except Exception as e:
        logger.error(f"Timezone normalization failed: {e}")
        return df
