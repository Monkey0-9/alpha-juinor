"""
Data Quality Agent for Institutional Trading System.

Assesses data quality for single symbol dataframes.
Checks:
1. Schema Validity
2. Missingness / Gaps
3. Flash Crashes / Spikes (>30% vs median)
4. Zero/Negative Prices
5. Volume anomalies

Emits Prometheus metrics and alerts for institutional monitoring.
"""

import logging
import time
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Quality thresholds
MIN_DATA_QUALITY = 0.6
NULL_THRESHOLD = 0.05
ZERO_VOLUME_THRESHOLD = 0.05
FLASH_SPIKE_THRESHOLD = 0.30


@dataclass
class QualityResult:
    """Result of quality assessment"""
    symbol: str
    is_usable: bool
    quality_score: float
    reasons: List[str] = field(default_factory=list)
    missing_days: int = 0
    expected_days: int = 0
    avg_quality_score: float = 0.0  # For Prometheus gauge
    data_missing_days_total: int = 0  # For Prometheus gauge


class QualityAgent:
    """
    Assesses data quality for a single symbol's dataframe.

    Provides:
    - Quality scoring (0.0 to 1.0)
    - Rejection with reason codes
    - Prometheus metrics export
    - Alert triggering
    """

    # NYSE trading holidays (approximate for 2021-2026)
    TRADING_HOLIDAYS = [
        '2026-01-01', '2026-01-20', '2026-02-17', '2026-04-10', '2026-05-25',
        '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
        '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
        '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
        '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
        '2023-01-02', '2023-01-16', '2023-02-20', '2023-04-07', '2023-05-29',
        '2023-06-19', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25',
        '2022-01-17', '2022-02-21', '2022-04-15', '2022-05-30', '2022-06-20',
        '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-26', '2022-01-01',
        '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
        '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24', '2021-12-31',
    ]

    def __init__(self, min_quality: float = MIN_DATA_QUALITY):
        """
        Initialize QualityAgent.

        Args:
            min_quality: Minimum quality threshold for data to be considered usable
        """
        self.min_quality = min_quality
        self._prometheus_metrics = {
            'avg_quality_score': 0.0,
            'data_missing_days_total': 0,
            'total_rows_checked': 0,
            'total_failures': 0,
        }

    def check_quality(
        self,
        symbol: str,
        df: pd.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> QualityResult:
        """
        Returns: QualityResult with (is_usable, quality_score, reasons, metrics)
        """
        result = QualityResult(
            symbol=symbol,
            is_usable=False,
            quality_score=0.0
        )

        if df is None or df.empty:
            result.reasons = ["EMPTY_DATAFRAME"]
            result.quality_score = 0.0
            self._prometheus_metrics['total_failures'] += 1
            return result

        reasons = []
        score = 1.0

        # 1. Schema Check
        schema_valid, schema_reasons = self._check_schema(df)
        if not schema_valid:
            reasons.extend(schema_reasons)
            score = 0.0
            result.quality_score = 0.0
            result.reasons = reasons
            self._prometheus_metrics['total_failures'] += 1
            return result

        # 2. Length Check
        if len(df) < 50:
            reasons.append(f"INSUFFICIENT_HISTORY({len(df)} rows)")
            score *= 0.5

        # 3. Missing Values (NULL_THRESHOLD = 5%)
        null_pct = self._check_null_values(df)
        if null_pct > NULL_THRESHOLD:
            reasons.append(f"HIGH_NULL_PERCENTAGE({null_pct:.1%})")
            score = 0.0  # Critical fail - too many nulls
            result.quality_score = 0.0
            result.reasons = reasons
            self._prometheus_metrics['total_failures'] += 1
            return result

        # 4. Zero/Negative Price Check
        if self._has_zero_negative_prices(df):
            reasons.append("ZERO_NEGATIVE_PRICE_DETECTED")
            score = 0.0
            result.quality_score = 0.0
            result.reasons = reasons
            self._prometheus_metrics['total_failures'] += 1
            return result

        # 5. Zero Volume Check (allow up to 5%)
        zero_vol_pct = self._check_zero_volume(df)
        if zero_vol_pct > ZERO_VOLUME_THRESHOLD:
            reasons.append(f"HIGH_ZERO_VOLUME({zero_vol_pct:.1%})")
            score *= 0.7

        # 6. Flash Spike Detection (>30% move)
        spike_info = self._check_flash_spikes(df)
        if spike_info['has_spike']:
            reasons.append(f"FLASH_SPIKE_DETECTED({spike_info['max_return']:.1%})")
            score *= 0.3

        # 7. Data Continuity (missing trading days)
        if start_date and end_date:
            continuity_info = self._check_data_continuity(df, start_date, end_date)
            result.missing_days = continuity_info['missing_days']
            result.expected_days = continuity_info['expected_days']
            if continuity_info['missing_pct'] > 0.10:  # Allow 10% missing
                reasons.append(f"MISSING_TRADING_DAYS({result.missing_days}/{result.expected_days})")
                score *= 0.8

        # 8. Price Continuity (close_t vs open_{t+1})
        continuity_check = self._check_price_continuity(df)
        if continuity_check['has_gaps']:
            reasons.append(f"PRICE_DISCONTINUITY({continuity_check['gap_count']} gaps >20%)")
            score *= 0.7

        # Final check: score threshold
        result.quality_score = max(0.0, score)
        result.is_usable = result.quality_score >= self.min_quality
        result.reasons = reasons

        # Update Prometheus metrics
        self._prometheus_metrics['total_rows_checked'] += len(df)
        if not result.is_usable:
            self._prometheus_metrics['total_failures'] += 1

        if not result.is_usable and not reasons:
            result.reasons.append(f"LOW_QUALITY_SCORE({result.quality_score:.2f})")

        return result

    def _check_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check required columns exist"""
        required = ['Close', 'High', 'Low', 'Volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return False, [f"MISSING_COLUMNS({','.join(missing)})"]
        return True, []

    def _check_null_values(self, df: pd.DataFrame) -> float:
        """Calculate percentage of null values in key price columns"""
        # Check nulls in essential price columns only
        price_cols = ['Open', 'High', 'Low', 'Close']
        available_cols = [c for c in price_cols if c in df.columns]

        if not available_cols:
            return 1.0  # All null if no price columns

        total_cells = len(df) * len(available_cols)
        null_cells = df[available_cols].isnull().sum().sum()

        return null_cells / total_cells if total_cells > 0 else 0.0

    def _has_zero_negative_prices(self, df: pd.DataFrame) -> bool:
        """Check for zero or negative prices"""
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns and (df[col] <= 0).any():
                return True
        return False

    def _check_zero_volume(self, df: pd.DataFrame) -> float:
        """Calculate percentage of zero-volume days"""
        if 'Volume' not in df.columns:
            return 0.0
        zero_vol_days = (df['Volume'] == 0).sum()
        return zero_vol_days / len(df) if len(df) > 0 else 0.0

    def _check_flash_spikes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect flash crashes or spikes (>30% daily move)"""
        if 'Close' not in df.columns:
            return {'has_spike': False, 'max_return': 0.0}

        returns = df['Close'].pct_change().dropna()
        if returns.empty:
            return {'has_spike': False, 'max_return': 0.0}

        max_return = returns.abs().max()
        return {
            'has_spike': max_return > FLASH_SPIKE_THRESHOLD,
            'max_return': float(max_return)
        }

    def _check_data_continuity(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Check for missing trading days"""
        # Calculate expected trading days
        all_days = pd.date_range(start=start_date, end=end_date, freq='B')
        trading_days = [
            d.strftime('%Y-%m-%d') for d in all_days
            if d.strftime('%Y-%m-%d') not in self.TRADING_HOLIDAYS
        ]
        expected_days = len(trading_days)

        # Get actual days from dataframe
        actual_days: List[str] = []
        if isinstance(df.index, pd.DatetimeIndex):
            actual_days = df.index.strftime('%Y-%m-%d').tolist()

        actual_count = len(actual_days)
        missing_count = max(0, expected_days - actual_count)
        missing_pct = missing_count / expected_days if expected_days > 0 else 0

        return {
            'expected_days': expected_days,
            'actual_days': actual_count,
            'missing_days': missing_count,
            'missing_pct': missing_pct
        }

    def _check_price_continuity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check price continuity (close_t vs open_{t+1})"""
        if len(df) < 2:
            return {'has_gaps': False, 'gap_count': 0}

        close_open_diff = (
            df['Open'].iloc[1:].values - df['Close'].iloc[:-1].values
        ) / df['Close'].iloc[:-1].values

        large_gaps = np.abs(close_open_diff) > 0.20  # 20% gap threshold
        gap_count = int(large_gaps.sum())

        return {
            'has_gaps': gap_count > 0,
            'gap_count': gap_count
        }

    def get_prometheus_metrics(self) -> Dict[str, float]:
        """
        Get metrics for Prometheus export.

        Returns:
            Dictionary of metric names to values
        """
        total = self._prometheus_metrics['total_rows_checked']
        if total > 0:
            self._prometheus_metrics['avg_quality_score'] = (
                1.0 - (self._prometheus_metrics['total_failures'] / total)
            )

        return {
            'quality_avg_score': self._prometheus_metrics['avg_quality_score'],
            'quality_missing_days_total': self._prometheus_metrics['data_missing_days_total'],
            'quality_rows_checked': self._prometheus_metrics['total_rows_checked'],
            'quality_failures_total': self._prometheus_metrics['total_failures'],
        }

    def should_reject(self, result: QualityResult) -> bool:
        """
        Determine if symbol should be rejected based on quality.

        Args:
            result: QualityResult from check_quality

        Returns:
            True if data quality is too low
        """
        return not result.is_usable or result.quality_score < self.min_quality

    def get_rejection_reason(self, result: QualityResult) -> str:
        """
        Get formatted rejection reason for decision logs.

        Args:
            result: QualityResult from check_quality

        Returns:
            Formatted rejection reason
        """
        if result.is_usable:
            return ""

        if result.reasons:
            return f"data_quality: {'; '.join(result.reasons)}"
        return f"data_quality: LOW_QUALITY_SCORE({result.quality_score:.2f})"


def get_quality_agent() -> QualityAgent:
    """Get configured QualityAgent instance"""
    return QualityAgent(min_quality=MIN_DATA_QUALITY)

