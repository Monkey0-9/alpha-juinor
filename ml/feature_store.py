
import logging
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime

from ml.feature_registry import get_feature_registry
from database.manager import DatabaseManager

logger = logging.getLogger("FEATURE_STORE")

class FeatureStore:
    """
    Centralized Feature Store for Point-in-Time Correctness.

    Ensures that features fetched for a specific time T do not
    contain future information.
    """

    def __init__(self):
        self.registry = get_feature_registry()
        self.db = DatabaseManager()

    def get_features(
        self,
        symbol: str,
        as_of_date: str,
        lookback_days: int = 252,
        feature_set_id: str = "standard_v1"
    ) -> pd.DataFrame:
        """
        Get point-in-time features for a symbol.

        Args:
            symbol: Ticker symbol
            as_of_date: Cutoff date (YYYY-MM-DD)
            lookback_days: Number of days of history to fetch prior to cutoff
            feature_set_id: ID of the feature set schema to use

        Returns:
            DataFrame with index (date) and feature columns.
        """
        # 1. Fetch raw price history up to as_of_date
        # Note: In a real PIT system, we'd query a PIT database.
        # Here we rely on the implementation ensuring no future leaks in calculation.

        raw_df = self.db.get_daily_prices(symbol, limit=lookback_days + 50) # Buffer for rolling calc
        if raw_df.empty:
            logger.warning(f"No data for {symbol}")
            return pd.DataFrame()

        # Ensure we don't return data past the cutoff (Critical for backtesting)
        # Note: get_daily_prices might return recent data. Filter it.
        raw_df = raw_df[raw_df['date'] <= as_of_date]

        # 2. Sort
        raw_df = raw_df.sort_values('date').set_index('date')

        # 3. Calculate features on the fly (Hybrid Approach)
        # In a fully matured system, features might be pre-computed and stored.
        # Here we compute dynamically using versioned logic.

        features_df = self._compute_standard_features(raw_df)

        # 4. Filter for requested schema if needed
        # (Simplified for now - returning all computed)

        return features_df.tail(lookback_days)

    def _compute_standard_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute standard technical features.

        Note: This duplicates some logic from `processors`, but is strictly versioned here.
        """
        # Ensure numeric
        for col in ['close', 'open', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Returns
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)

        # Volatility
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()

        # Volume Ratio
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        # Momentum
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1.0

        # Clean infinite/NaN
        df = df.replace([float('inf'), float('-inf')], float('nan')).fillna(0.0)

        return df

# Singleton
_store_instance = None

def get_feature_store() -> FeatureStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = FeatureStore()
    return _store_instance
