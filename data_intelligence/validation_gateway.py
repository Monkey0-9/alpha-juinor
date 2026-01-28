"""
data_intelligence/validation_gateway.py

Central Data Validation Gateway (Ticket 3)

Single entry point for ALL data validation.
Runs comprehensive checks and computes DATA_QUALITY_SCORE.
Never drops data silently - stores validation flags and preserves raw data.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger("VALIDATION_GATEWAY")


class ValidationFlag(str, Enum):
    """Validation issue flags."""
    MISSING_DATES = "MISSING_DATES"
    DUPLICATE_DATES = "DUPLICATE_DATES"
    ZERO_PRICE = "ZERO_PRICE"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    PRICE_SPIKE = "PRICE_SPIKE"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    CHRONOLOGY_ERROR = "CHRONOLOGY_ERROR"
    STALE_DATA = "STALE_DATA"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    MISSING_COLUMNS = "MISSING_COLUMNS"
    NULL_VALUES = "NULL_VALUES"
    ADJUSTED_CLOSE_MISSING = "ADJUSTED_CLOSE_MISSING"


@dataclass
class ValidationResult:
    """Result of data validation."""
    symbol: str
    is_valid: bool
    quality_score: float  # [0.0, 1.0]
    flags: List[ValidationFlag] = field(default_factory=list)
    flag_details: Dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    checked_at: str = ""
    raw_hash: str = ""  # Hash of raw data for provenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "flags": [f.value for f in self.flags],
            "flag_details": self.flag_details,
            "row_count": self.row_count,
            "checked_at": self.checked_at,
            "raw_hash": self.raw_hash
        }


class ValidationGateway:
    """
    Central Data Validation Gateway.

    All data must pass through this gateway before being used.
    Computes quality scores and validation flags.

    Quality Score Components:
    - Completeness: 30% (missing dates, null values)
    - Accuracy: 30% (spikes, zero prices)
    - Consistency: 20% (duplicates, chronology)
    - Freshness: 20% (staleness)

    Thresholds:
    - quality >= 0.9: OK
    - quality >= 0.6: DEGRADED_DATA
    - quality < 0.6: INVALID_DATA
    """

    # Required columns
    REQUIRED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']
    OPTIONAL_COLUMNS = ['adjusted_close', 'provider', 'symbol']

    # Thresholds
    SPIKE_THRESHOLD = 0.30  # 30% daily move = spike
    VOLUME_SPIKE_THRESHOLD = 10.0  # 10x median volume = spike
    MAX_NULL_RATIO = 0.05  # 5% null values max
    STALENESS_HOURS = 48  # Data older than this = stale
    MIN_HISTORY_DAYS = 252  # Minimum 1 year of data

    def __init__(self, db_manager=None):
        """
        Initialize ValidationGateway.

        Args:
            db_manager: DatabaseManager instance (lazy loaded if None)
        """
        self._db = db_manager

    @property
    def db(self):
        if self._db is None:
            from database.manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    def validate(
        self,
        symbol: str,
        df: pd.DataFrame,
        expected_start: Optional[str] = None,
        expected_end: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a DataFrame of price data.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            expected_start: Expected start date (for gap detection)
            expected_end: Expected end date

        Returns:
            ValidationResult with quality score and flags
        """
        now = datetime.utcnow().isoformat() + 'Z'
        flags: List[ValidationFlag] = []
        flag_details: Dict[str, Any] = {}

        # Compute raw hash for provenance
        raw_hash = self._compute_hash(df)

        # Initialize score components
        completeness_score = 1.0
        accuracy_score = 1.0
        consistency_score = 1.0
        freshness_score = 1.0

        # ========== COMPLETENESS CHECKS ==========

        # 1. Check required columns
        missing_cols = self._check_missing_columns(df)
        if missing_cols:
            flags.append(ValidationFlag.MISSING_COLUMNS)
            flag_details['missing_columns'] = missing_cols
            completeness_score *= 0.5

        # 2. Check for adjusted_close (warning only)
        if 'adjusted_close' not in df.columns:
            flags.append(ValidationFlag.ADJUSTED_CLOSE_MISSING)
            flag_details['adjusted_close_warning'] = "No adjusted_close column - using close"

        # 3. Check null values
        null_ratio = self._check_null_values(df)
        if null_ratio > self.MAX_NULL_RATIO:
            flags.append(ValidationFlag.NULL_VALUES)
            flag_details['null_ratio'] = null_ratio
            completeness_score *= (1 - null_ratio)

        # 4. Check missing dates
        if expected_start and expected_end:
            missing_dates_info = self._check_missing_dates(df, expected_start, expected_end)
            if missing_dates_info['missing_pct'] > 0.05:
                flags.append(ValidationFlag.MISSING_DATES)
                flag_details['missing_dates'] = missing_dates_info
                completeness_score *= (1 - missing_dates_info['missing_pct'])

        # 5. Check minimum history
        if len(df) < self.MIN_HISTORY_DAYS:
            flags.append(ValidationFlag.INSUFFICIENT_HISTORY)
            flag_details['history_days'] = len(df)
            flag_details['required_days'] = self.MIN_HISTORY_DAYS
            completeness_score *= (len(df) / self.MIN_HISTORY_DAYS)

        # ========== ACCURACY CHECKS ==========

        # 6. Check zero/negative prices
        zero_neg_info = self._check_zero_negative_prices(df)
        if zero_neg_info['zero_count'] > 0:
            flags.append(ValidationFlag.ZERO_PRICE)
            flag_details['zero_prices'] = zero_neg_info['zero_count']
            accuracy_score *= 0.8
        if zero_neg_info['negative_count'] > 0:
            flags.append(ValidationFlag.NEGATIVE_PRICE)
            flag_details['negative_prices'] = zero_neg_info['negative_count']
            accuracy_score *= 0.5  # Negative prices are critical

        # 7. Check price spikes
        spike_info = self._check_price_spikes(df)
        if spike_info['spike_count'] > 0:
            flags.append(ValidationFlag.PRICE_SPIKE)
            flag_details['price_spikes'] = spike_info
            # Penalize based on spike severity
            accuracy_score *= max(0.5, 1 - (spike_info['spike_count'] / len(df)))

        # 8. Check volume spikes
        vol_spike_info = self._check_volume_spikes(df)
        if vol_spike_info['spike_count'] > 5:  # Some volume spikes are normal
            flags.append(ValidationFlag.VOLUME_SPIKE)
            flag_details['volume_spikes'] = vol_spike_info

        # ========== CONSISTENCY CHECKS ==========

        # 9. Check duplicates
        dup_info = self._check_duplicates(df)
        if dup_info['duplicate_count'] > 0:
            flags.append(ValidationFlag.DUPLICATE_DATES)
            flag_details['duplicates'] = dup_info
            consistency_score *= (1 - dup_info['duplicate_count'] / max(1, len(df)))

        # 10. Check chronology
        chron_info = self._check_chronology(df)
        if not chron_info['is_sorted']:
            flags.append(ValidationFlag.CHRONOLOGY_ERROR)
            flag_details['chronology'] = chron_info
            consistency_score *= 0.8

        # ========== FRESHNESS CHECK ==========

        # 11. Check staleness
        stale_info = self._check_staleness(df)
        if stale_info['is_stale']:
            flags.append(ValidationFlag.STALE_DATA)
            flag_details['staleness'] = stale_info
            freshness_score *= max(0.3, 1 - (stale_info['hours_old'] / (self.STALENESS_HOURS * 4)))

        # ========== COMPUTE FINAL SCORE ==========

        quality_score = (
            0.30 * completeness_score +
            0.30 * accuracy_score +
            0.20 * consistency_score +
            0.20 * freshness_score
        )
        quality_score = round(max(0.0, min(1.0, quality_score)), 4)

        # Determine validity
        is_valid = quality_score >= 0.6 and ValidationFlag.NEGATIVE_PRICE not in flags

        result = ValidationResult(
            symbol=symbol,
            is_valid=is_valid,
            quality_score=quality_score,
            flags=flags,
            flag_details=flag_details,
            row_count=len(df),
            checked_at=now,
            raw_hash=raw_hash
        )

        # Log result
        logger.info(json.dumps({
            "event": "VALIDATION_COMPLETE",
            "symbol": symbol,
            "quality_score": quality_score,
            "is_valid": is_valid,
            "flags": [f.value for f in flags],
            "row_count": len(df)
        }))

        return result

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute MD5 hash of DataFrame for provenance."""
        try:
            # Hash first 100 rows and last 10 rows for efficiency
            sample = pd.concat([df.head(100), df.tail(10)])
            content = sample.to_json(orient='records')
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return ""

    def _check_missing_columns(self, df: pd.DataFrame) -> List[str]:
        """Check for missing required columns."""
        return [col for col in self.REQUIRED_COLUMNS if col not in df.columns]

    def _check_null_values(self, df: pd.DataFrame) -> float:
        """Calculate ratio of null values in critical columns."""
        critical_cols = [c for c in ['close', 'open', 'high', 'low'] if c in df.columns]
        if not critical_cols:
            return 1.0

        total_cells = len(df) * len(critical_cols)
        null_cells = df[critical_cols].isnull().sum().sum()
        return null_cells / max(1, total_cells)

    def _check_missing_dates(
        self,
        df: pd.DataFrame,
        expected_start: str,
        expected_end: str
    ) -> Dict[str, Any]:
        """Check for missing trading dates."""
        try:
            if 'date' not in df.columns:
                return {'missing_pct': 0, 'missing_dates': []}

            # Parse dates
            df_dates = pd.to_datetime(df['date'])
            start = pd.to_datetime(expected_start)
            end = pd.to_datetime(expected_end)

            # Generate expected trading days (weekdays)
            all_days = pd.date_range(start=start, end=end, freq='B')  # Business days

            actual_dates = set(df_dates.dt.date)
            expected_dates = set(d.date() for d in all_days)

            missing = expected_dates - actual_dates
            missing_pct = len(missing) / max(1, len(expected_dates))

            return {
                'missing_pct': round(missing_pct, 4),
                'missing_count': len(missing),
                'expected_count': len(expected_dates),
                'sample_missing': [str(d) for d in list(missing)[:5]]
            }
        except Exception as e:
            logger.warning(f"Error checking missing dates: {e}")
            return {'missing_pct': 0, 'error': str(e)}

    def _check_zero_negative_prices(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for zero or negative prices."""
        price_cols = [c for c in ['close', 'open', 'high', 'low'] if c in df.columns]

        zero_count = 0
        negative_count = 0

        for col in price_cols:
            zero_count += (df[col] == 0).sum()
            negative_count += (df[col] < 0).sum()

        return {
            'zero_count': int(zero_count),
            'negative_count': int(negative_count)
        }

    def _check_price_spikes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect abnormal price spikes (>30% daily move)."""
        if 'close' not in df.columns or len(df) < 2:
            return {'spike_count': 0, 'max_spike': 0.0}

        try:
            returns = df['close'].pct_change().abs()
            spikes = returns > self.SPIKE_THRESHOLD
            spike_count = spikes.sum()
            max_spike = returns.max()

            # Get spike dates
            spike_indices = df.index[spikes]
            spike_details = []
            if 'date' in df.columns:
                for idx in spike_indices[:5]:  # First 5 spikes
                    spike_details.append({
                        'date': str(df.loc[idx, 'date']) if idx in df.index else 'unknown',
                        'return': round(returns.loc[idx] if idx in returns.index else 0, 4)
                    })

            return {
                'spike_count': int(spike_count),
                'max_spike': round(float(max_spike), 4) if not pd.isna(max_spike) else 0.0,
                'threshold': self.SPIKE_THRESHOLD,
                'sample_spikes': spike_details
            }
        except Exception as e:
            logger.warning(f"Error checking price spikes: {e}")
            return {'spike_count': 0, 'max_spike': 0.0, 'error': str(e)}

    def _check_volume_spikes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect abnormal volume spikes."""
        if 'volume' not in df.columns or len(df) < 20:
            return {'spike_count': 0}

        try:
            median_vol = df['volume'].median()
            if median_vol > 0:
                vol_ratio = df['volume'] / median_vol
                spike_count = (vol_ratio > self.VOLUME_SPIKE_THRESHOLD).sum()
                max_ratio = vol_ratio.max()
            else:
                spike_count = 0
                max_ratio = 0.0

            return {
                'spike_count': int(spike_count),
                'max_ratio': round(float(max_ratio), 2) if not pd.isna(max_ratio) else 0.0,
                'threshold': self.VOLUME_SPIKE_THRESHOLD
            }
        except Exception as e:
            return {'spike_count': 0, 'error': str(e)}

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate dates."""
        if 'date' not in df.columns:
            return {'duplicate_count': 0}

        dup_count = df['date'].duplicated().sum()
        return {
            'duplicate_count': int(dup_count),
            'sample_duplicates': df[df['date'].duplicated()]['date'].head(5).tolist()
        }

    def _check_chronology(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if dates are in chronological order."""
        if 'date' not in df.columns:
            return {'is_sorted': True}

        try:
            dates = pd.to_datetime(df['date'])
            is_sorted = dates.is_monotonic_increasing

            if not is_sorted:
                # Find out-of-order indices
                wrong_order = []
                for i in range(1, min(len(dates), 100)):
                    if dates.iloc[i] < dates.iloc[i-1]:
                        wrong_order.append(i)
                return {
                    'is_sorted': False,
                    'out_of_order_indices': wrong_order[:5]
                }

            return {'is_sorted': True}
        except Exception as e:
            return {'is_sorted': True, 'error': str(e)}

    def _check_staleness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if data is stale."""
        if 'date' not in df.columns or len(df) == 0:
            return {'is_stale': True, 'hours_old': 999}

        try:
            last_date = pd.to_datetime(df['date'].max())
            now = datetime.utcnow()

            # Adjust for timezone-naive
            if last_date.tzinfo:
                last_date = last_date.tz_localize(None)

            hours_old = (now - last_date).total_seconds() / 3600
            is_stale = hours_old > self.STALENESS_HOURS

            return {
                'is_stale': is_stale,
                'hours_old': round(hours_old, 1),
                'last_date': str(last_date.date()),
                'threshold_hours': self.STALENESS_HOURS
            }
        except Exception as e:
            return {'is_stale': True, 'error': str(e)}

    def validate_and_score(
        self,
        symbol: str,
        df: pd.DataFrame,
        persist: bool = True
    ) -> Tuple[ValidationResult, float]:
        """
        Validate data and return both result and score.

        Convenience method that also integrates with data state machine.

        Args:
            symbol: Stock symbol
            df: DataFrame to validate
            persist: Whether to persist to data_state and data_quality tables

        Returns:
            Tuple of (ValidationResult, quality_score)
        """
        result = self.validate(symbol, df)

        if persist:
            try:
                # Update data state machine
                from data_intelligence.data_state_machine import get_data_state_machine

                dsm = get_data_state_machine()
                last_date = df['date'].max() if 'date' in df.columns else datetime.utcnow().isoformat()

                dsm.evaluate_and_transition(
                    symbol=symbol,
                    quality_score=result.quality_score,
                    data_timestamp=str(last_date),
                    provider=df.get('provider', ['unknown'])[0] if 'provider' in df.columns else 'unknown',
                    validation_passed=result.is_valid
                )
            except Exception as e:
                logger.warning(f"Failed to update data state for {symbol}: {e}")

        return result, result.quality_score


# Singleton instance
_instance: Optional[ValidationGateway] = None


def get_validation_gateway() -> ValidationGateway:
    """Get singleton ValidationGateway instance."""
    global _instance
    if _instance is None:
        _instance = ValidationGateway()
    return _instance
