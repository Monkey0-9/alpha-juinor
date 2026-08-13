import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def check_timestamp_monotonicity(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Ensures that timestamps for each symbol are strictly increasing.
    """
    if df.empty or "timestamp" not in df.columns or "symbol" not in df.columns:
        return True, df

    passed = True
    bad_rows = []

    for symbol, group in df.groupby("symbol"):
        if not group["timestamp"].is_monotonic_increasing:
            passed = False
            # Find rows where time goes backwards
            diffs = group["timestamp"].diff()
            bad = group[diffs < pd.Timedelta(0)]
            bad_rows.append(bad)
            logger.warning(f"Symbol {symbol} has non-monotonic timestamps.")

    if not passed:
        bad_df = pd.concat(bad_rows)
        # Drop bad rows to clean the dataframe
        clean_df = df.drop(bad_df.index)
        return False, clean_df

    return True, df


def check_duplicates(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Detects and removes exact duplicate rows based on symbol and timestamp.
    """
    if df.empty or "timestamp" not in df.columns or "symbol" not in df.columns:
        return True, df

    dup_mask = df.duplicated(subset=["symbol", "timestamp"], keep="first")
    if dup_mask.any():
        logger.warning(f"Found {dup_mask.sum()} duplicate timestamps. Deduplicating.")
        clean_df = df[~dup_mask].copy()
        return False, clean_df

    return True, df


def check_missing_values(
    df: pd.DataFrame, threshold: float = 0.05
) -> Tuple[bool, pd.DataFrame]:
    """
    Checks if critical columns have more than threshold % of missing values.
    Fills short gaps and drops rows if gap is too large.
    """
    if df.empty:
        return True, df

    passed = True
    cols_to_check = ["open", "high", "low", "close", "volume"]

    clean_df = df.copy()

    for col in cols_to_check:
        if col in clean_df.columns:
            missing_pct = clean_df[col].isna().mean()
            if missing_pct > threshold:
                logger.error(
                    f"Column {col} has {missing_pct:.2%} missing values (threshold {threshold:.2%})."
                )
                passed = False

            # Forward fill up to 3 periods for small gaps
            clean_df[col] = clean_df.groupby("symbol")[col].ffill(limit=3)

    # Drop rows that are still NaN in close price
    clean_df = clean_df.dropna(subset=["close"])

    if len(clean_df) < len(df):
        logger.warning(
            f"Dropped {len(df) - len(clean_df)} rows due to unfillable missing values."
        )
        passed = False

    return passed, clean_df


def check_negative_prices(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Ensures prices are not negative (unless specified for strange assets, but typically disallowed).
    """
    if df.empty:
        return True, df

    cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]

    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask = mask | (df[col] < 0)

    if mask.any():
        logger.error(f"Found {mask.sum()} rows with negative prices. Dropping them.")
        return False, df[~mask].copy()

    return True, df
