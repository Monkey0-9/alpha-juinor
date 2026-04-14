"""
Institutional Data Quality & Validation Engine.

Responsibilities:
1. Standardized data validation routine.
2. Calculation of DATA_QUALITY_SCORE (0.0 to 1.0).
3. Flagging of specific data issues (Spikes, Gaps, Zeros).
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

logger = logging.getLogger("DATA_QUALITY")

class DataQualityEngine:
    """
    Centralized validation engine for all market data ingestion.
    """

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, symbol: str) -> Tuple[float, Dict[str, Any]]:
        """
        Validate OHLCV DataFrame.

        Checks:
        1. Missing Dates (Gaps > 4 days for business days, though simpler gap check used here).
        2. Duplicates.
        3. Zero/Negative Prices.
        4. Chronological Order.
        5. Extreme Spikes (>50% day-over-day range or high/low mismatch).

        Returns:
            (quality_score, details_dict)
            details_dict contains {'flags': [], 'status': 'SUCCESS'|'FAILED'}
        """
        if df is None or df.empty:
            return 0.0, {"flags": ["EMPTY_DATA"], "status": "FAILED"}

        flags = []
        scores = []

        # ---------------------------------------------------------------------
        # 1. Missing Dates / Gaps
        # ---------------------------------------------------------------------
        if len(df) > 1:
            # Ensure DateTime index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    return 0.0, {"flags": ["INVALID_INDEX"], "status": "FAILED"}

            df = df.sort_index()
            diffs = df.index.to_series().diff().dt.days

            # Allow for weekends (3 days), holidays. greater than 4 is suspicious for stocks.
            # For Crypto (24/7), > 1 is suspicious.
            # We'll use a conservative > 5 threshold for generalized "gap" warning to avoid noise
            gap_threshold = 5
            gaps = diffs[diffs > gap_threshold]

            if not gaps.empty:
                count = len(gaps)
                flags.append(f"MISSING_DATES_GAPS_{count}")
                # Penalty: 10% per gap found
                scores.append(max(0.0, 1.0 - (count * 0.1)))
            else:
                scores.append(1.0)
        else:
            # Single row is barely usable history
            scores.append(0.5)

        # ---------------------------------------------------------------------
        # 2. Duplicates
        # ---------------------------------------------------------------------
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            flags.append(f"DUPLICATE_ROWS_{duplicates}")
            scores.append(max(0.0, 1.0 - (duplicates / len(df))))
        else:
            scores.append(1.0)

        # ---------------------------------------------------------------------
        # 3. Zero / Negative Prices
        # ---------------------------------------------------------------------
        cols = ['Open', 'High', 'Low', 'Close']
        available_cols = [c for c in cols if c in df.columns]

        if available_cols:
            invalid_prices = (df[available_cols] <= 0).any(axis=1).sum()
            if invalid_prices > 0:
                flags.append(f"ZERO_NEGATIVE_PRICES_{invalid_prices}")
                scores.append(0.0) # Critical failure
            else:
                scores.append(1.0)

        # ---------------------------------------------------------------------
        # 4. Extreme Spikes / Anomalies
        # ---------------------------------------------------------------------
        # Check High < Low (Impossible)
        if 'High' in df.columns and 'Low' in df.columns:
            inverted = (df['High'] < df['Low']).sum()
            if inverted > 0:
                flags.append(f"HIGH_LESS_THAN_LOW_{inverted}")
                scores.append(0.0)
            else:
                scores.append(1.0)

            # Check Extreme Range (High is 2x Low is very rare for heavy liquid stocks, but possible in penny/crypto)
            # We flag > 100% intra-day move as warning
            range_pct = (df['High'] - df['Low']) / df['Low']
            extreme_moves = (range_pct > 1.0).sum()
            if extreme_moves > 0:
                flags.append(f"EXTREME_VOLATILITY_{extreme_moves}")
                # Soft penalty
                scores.append(0.9)

        # ---------------------------------------------------------------------
        # 5. Chronological Order
        # ---------------------------------------------------------------------
        if not df.index.is_monotonic_increasing:
            flags.append("CHRONOLOGICAL_ERROR")
            scores.append(0.0)
        else:
            scores.append(1.0)

        # ---------------------------------------------------------------------
        # Final Scoring
        # ---------------------------------------------------------------------
        # Weighting:
        # Zeros/Negatives/Inverted = Kill score (append 0.0 above)
        # Gaps = Reduce score

        if not scores:
            final_score = 1.0
        else:
            if 0.0 in scores:
                final_score = 0.0
            else:
                final_score = np.mean(scores)

        status = "SUCCESS" if final_score >= 0.6 else "INVALID_DATA"

        details = {
            "flags": flags,
            "status": status,
            "sub_scores": scores
        }

        return float(final_score), details
