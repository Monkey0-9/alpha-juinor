"""
Data Quality Scoring Engine

Computes data quality scores for price history data and identifies issues.
Used by ingestion pipeline and symbol classification system.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def compute_data_quality(df: pd.DataFrame) -> Tuple[float, List[str]]:
    """
    Compute data quality score (0.0 to 1.0) and identify issues.

    Args:
        df: DataFrame with columns: date/index, close, volume (optional)

    Returns:
        Tuple of (quality_score, reason_codes)

    Quality Checks:
    - Missing values in critical columns
    - Duplicate dates
    - Zero or negative prices
    - Extreme volume spikes (>99.9th percentile)
    """
    if df is None or len(df) == 0:
        return 0.0, ["NO_DATA"]

    reasons = []
    score = 1.0

    # Normalize column names (handle both 'close' and 'Close')
    df_check = df.copy()
    if 'Close' in df_check.columns:
        df_check['close'] = df_check['Close']
    if 'Volume' in df_check.columns:
        df_check['volume'] = df_check['Volume']

    # Check 1: Missing values in close price
    if 'close' not in df_check.columns:
        return 0.0, ["MISSING_CLOSE_COLUMN"]

    missing_pct = df_check['close'].isna().mean()
    if missing_pct > 0:
        score -= missing_pct * 0.3
        reasons.append(f"MISSING_VALUES:{missing_pct:.1%}")

    # Check 2: Duplicate dates
    if isinstance(df_check.index, pd.DatetimeIndex):
        dup_pct = df_check.index.duplicated().mean()
    else:
        # If there's a date column
        if 'date' in df_check.columns:
            dup_pct = df_check.duplicated(subset=['date']).mean()
        else:
            dup_pct = 0.0

    if dup_pct > 0:
        score -= dup_pct * 0.2
        reasons.append(f"DUPLICATE_DATES:{dup_pct:.1%}")

    # Check 3: Zero or negative prices
    zero_or_negative = (df_check['close'] <= 0).any()
    if zero_or_negative:
        score -= 0.2
        reasons.append("ZERO_OR_NEGATIVE_PRICES")

    # Check 4: Extreme volume spikes (if volume available)
    if 'volume' in df_check.columns and df_check['volume'].notnull().any():
        vol_data = df_check['volume'].dropna()
        if len(vol_data) > 10:
            vol_threshold = vol_data.quantile(0.99) # Institutional: 99th percentile
            extreme_vol_pct = (vol_data > vol_threshold).sum() / max(1, len(vol_data))

            if extreme_vol_pct > 0.05: # More than 5% of data points exceeding 99th percentile implies structural issues
                score -= 0.15
                reasons.append(f"ERRATIC_VOLUME:{extreme_vol_pct:.1%}")

    # Check 5: Price Gaps / Spikes (Institutional Continuity Check)
    # 10% daily jump without volume expansion is highly suspicious
    if len(df_check) > 1:
        returns = df_check['close'].pct_change().abs()
        # Log-returns for more symmetrical spike detection
        log_rets = np.log(df_check['close'] / df_check['close'].shift(1)).abs()

        if 'volume' in df_check.columns:
            # rolling window for average volume
            avg_vol = df_check['volume'].rolling(20).mean()
            vol_rel = df_check['volume'] / avg_vol
            suspicious_jumps = ((returns > 0.10) & (vol_rel < 1.0)).sum()
        else:
            suspicious_jumps = (returns > 0.15).sum() # Higher threshold if no volume data

        if suspicious_jumps > 0:
            score -= min(0.3, suspicious_jumps * 0.05)
            reasons.append(f"UNEXPLAINED_PRICE_SPIKES:{suspicious_jumps}")

    # Check 6: Business Day Continuity (Institutional Grade)
    if isinstance(df_check.index, pd.DatetimeIndex) and len(df_check) > 5:
        # Check for missing business days if index is supposed to be daily
        expected_range = pd.bdate_range(start=df_check.index.min(), end=df_check.index.max())
        missing_days = len(expected_range) - len(df_check.index.unique())
        if missing_days > 5: # Allow some holidays/misses
             missing_pct = missing_days / len(expected_range)
             score -= min(0.2, missing_pct)
             reasons.append(f"MISSING_CONTINUITY:{missing_days}_days")

    # Check 7: Price Monotonicity / Split logic proxy
    # If unadjusted 'Close' drops significantly but 'Adj Close' (if present) doesn't,
    # or if we see a 50% drop/jump without news/volume, it's a split issue.
    if 'Adj Close' in df_check.columns and 'Close' in df_check.columns:
         ratio = df_check['Close'] / df_check['Adj Close']
         # Ratio should be non-decreasing (except for corporate actions we don't handle)
         if not ratio.is_monotonic_increasing:
              # We don't strictly penalize but flag for audit if many reversals
              reversals = (ratio.diff() < -0.01).sum()
              if reversals > 0:
                  score -= 0.1
                  reasons.append(f"ADJUSTMENT_INCONSISTENCY:{reversals}")

    # Check 8: Data chronology (if datetime index)
    if isinstance(df_check.index, pd.DatetimeIndex):
        if not df_check.index.is_monotonic_increasing:
            score -= 0.1
            reasons.append("NON_CHRONOLOGICAL_DATA")

    # Ensure score is in valid range
    score = max(0.0, min(1.0, score))

    if not reasons:
        reasons.append("OK")

    return score, reasons


def validate_data_for_trading(df: pd.DataFrame, min_rows: int = 1260, min_quality: float = 0.6) -> Tuple[bool, str]:
    """
    Validate if data is suitable for trading.

    Args:
        df: Price history DataFrame
        min_rows: Minimum required rows (default: 1260 = ~5 years)
        min_quality: Minimum quality score (default: 0.6)

    Returns:
        Tuple of (is_valid, reason)
    """
    if df is None or len(df) == 0:
        return False, "NO_DATA"

    # Check row count
    if len(df) < min_rows:
        return False, f"INSUFFICIENT_HISTORY:rows={len(df)}<{min_rows}"

    # Check quality
    quality_score, reason_codes = compute_data_quality(df)

    if quality_score < min_quality:
        return False, f"LOW_QUALITY:score={quality_score:.2f}<{min_quality}:reasons={','.join(reason_codes)}"

    return True, "OK"


def validate_data_for_ml(df: pd.DataFrame, min_rows: int = 1260, min_quality: float = 0.7) -> Tuple[bool, List[str]]:
    """
    Stricter validation for ML training.

    Args:
        df: Price history DataFrame
        min_rows: Minimum required rows (default: 1260)
        min_quality: Minimum quality score (default: 0.7 - stricter than trading)

    Returns:
        Tuple of (is_ready, reason_codes)
    """
    reasons = []

    if df is None or len(df) == 0:
        return False, ["NO_DATA"]

    # Check row count
    if len(df) < min_rows:
        reasons.append(f"INSUFFICIENT_SAMPLES:rows={len(df)}<{min_rows}")

    # Check quality with stricter threshold
    quality_score, quality_reasons = compute_data_quality(df)

    if quality_score < min_quality:
        reasons.append(f"INSUFFICIENT_QUALITY:score={quality_score:.2f}<{min_quality}")
        reasons.extend(quality_reasons)

    # Check for required columns
    required_cols = ['Close']
    missing_cols = [col for col in required_cols if col not in df.columns and col.lower() not in df.columns]
    if missing_cols:
        reasons.append(f"MISSING_COLUMNS:{','.join(missing_cols)}")

    is_ready = len(reasons) == 0

    if not reasons:
        reasons.append("ML_READY")

    return is_ready, reasons
