import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import json

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Ruthless Institutional Data Validator.
    Enforces strict survival-focused rules and computes DATA_QUALITY_SCORE.
    """

    def __init__(self, spike_threshold: float = 0.5, volume_z_threshold: float = 6.0):
        self.spike_threshold = spike_threshold
        self.volume_z_threshold = volume_z_threshold

    def validate_symbol_data(self, symbol: str, df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """
        Validate data for a single symbol and return score + summary.
        """
        if df is None or df.empty:
            return 0.0, {"status": "INVALID_DATA", "reason": "EMPTY_OR_NONE"}

        issues = []
        flags = {}

        # 1. Chronological order
        if not df.index.is_monotonic_increasing:
            issues.append("NOT_CHRONOLOGICAL")
            df = df.sort_index()

        # 2. Duplicate rows
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            issues.append(f"DUPLICATE_ROWS_{duplicates}")
            df = df[~df.index.duplicated(keep='first')]

        # 3. Missing dates (Gaps)
        # Assuming we expect business days. For simplicity, we check frequency if possible.
        # But institutional requirement is specific about 2 consecutive trading days.
        # We'll check the diff in index.
        date_diffs = df.index.to_series().diff().dt.days
        max_gap = date_diffs.max() if not date_diffs.empty else 0
        gap_penalty = 0.0
        if max_gap > 2:
            issues.append(f"MAX_GAP_{max_gap}_DAYS")
            gap_penalty = 0.4 # Heavy penalty (completeness is 40%)

        # 4. Zero/Negative prices
        invalid_bars = ((df['Open'] <= 0) | (df['High'] <= 0) | (df['Low'] <= 0) | (df['Close'] <= 0)).sum()
        if invalid_bars > 0:
            issues.append(f"INVALID_BARS_{invalid_bars}")

        # 5. Extreme price spikes (>50%)
        # Using Adjusted Close if available, else Close
        price_col = 'adjusted_close' if 'adjusted_close' in df.columns else 'Close'
        price_pct_change = df[price_col].pct_change().abs()
        extreme_spikes = (price_pct_change > self.spike_threshold).sum()
        if extreme_spikes > 0:
            issues.append(f"EXTREME_SPIKES_{extreme_spikes}")

        # 6. Abnormal volume spikes (z-score > 6 on 60-day rolling)
        volume_mean = df['Volume'].rolling(window=60).mean()
        volume_std = df['Volume'].rolling(window=60).std()
        df['vol_z'] = (df['Volume'] - volume_mean) / volume_std
        volume_anomalies = (df['vol_z'].abs() > self.volume_z_threshold).sum()
        if volume_anomalies > 0:
            issues.append(f"VOLUME_ANOMALIES_{volume_anomalies}")

        # WEIGHTED SCORE COMPUTATION
        # completeness (40%), no-invalid-bars (30%), volume sanity (10%), corporate-action consistency (10%), provider reliability (10%)
        # For this single run check:
        # Completeness: 1.0 - gap_penalty
        # Invalid Bars: 1.0 - (invalid_bars / len(df)) * 3 (aggressive)
        # Volume sanity: 1.0 - (volume_anomalies / len(df)) * 5

        comp_score = max(0.0, 1.0 - (gap_penalty))
        inv_bar_score = max(0.0, 1.0 - (invalid_bars / len(df) if len(df) > 0 else 1))
        if extreme_spikes > 0: inv_bar_score *= 0.5 # Double penalty for spikes

        vol_score = max(0.0, 1.0 - (volume_anomalies / len(df) if len(df) > 0 else 1))

        final_score = (comp_score * 0.4) + (inv_bar_score * 0.3) + (vol_score * 0.1) + (0.9 * 0.2) # Default 0.9 for corp/provider for now

        summary = {
            "status": "OK" if final_score >= 0.8 else "DEGRADED" if final_score >= 0.6 else "INVALID_DATA",
            "score": round(final_score, 4),
            "issues": issues,
            "max_gap": int(max_gap),
            "invalid_bars": int(invalid_bars),
            "extreme_spikes": int(extreme_spikes),
            "volume_anomalies": int(volume_anomalies),
            "row_count": len(df)
        }

        return final_score, summary
