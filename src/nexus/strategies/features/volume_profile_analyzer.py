"""
Volume Profile Analyzer
=======================

Calculates Volume Profile metrics:
- Point of Control (POC): Price level with highest traded volume.
- Value Area (VA): Price range containing 70% of total volume.
- High/Low Volume Nodes (HVN/LVN): Local maxima/minima in the volume histogram.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class VolumeProfileAnalyzer:
    def __init__(self, n_bins: int = 50, value_area_pct: float = 0.70):
        self.n_bins = n_bins
        self.value_area_pct = value_area_pct

        # Internal state
        self.profiles: Dict[str, pd.DataFrame] = {} # Symbol -> Histogram

    def update(self, market_data: pd.DataFrame):
        """
        Update volume profiles with new market data.
        Expects keys: 'Close', 'Volume'.
        For best results, pass high-frequency bars or tick data.
        """
        # Logic: We treat the passed DataFrame as a window of recent data
        # We bin the High-Low range or just use Close levels weighted by Volume

        # 1. Inspect Data structure
        # Assume MultiIndex (Symbol, Index) or Panel?
        # Or simple DataFrame with 'Close' and 'Volume' if single symbol.
        # Let's standardize on standard Pandas format where columns might be MultiIndex or we iterate symbols.

        symbols = []
        if isinstance(market_data.columns, pd.MultiIndex):
            symbols = market_data.columns.get_level_values(0).unique()
        elif 'Close' in market_data.columns:
            # Single symbol frame, but we need a symbol identifier.
            # We'll handle it if passed explicitly, but for now assuming MultiIndex for portfolio.
            pass

        for symbol in symbols:
            try:
                df = market_data[symbol].dropna()
                if df.empty or 'Close' not in df.columns or 'Volume' not in df.columns:
                    continue

                self._calculate_profile(symbol, df)
            except Exception as e:
                logger.error(f"Error calculating VP for {symbol}: {e}")

    def _calculate_profile(self, symbol: str, data: pd.DataFrame):
        """Calculate histogram and key levels."""
        prices = data['Close']
        volumes = data['Volume']

        if len(prices) == 0:
            return

        min_px = prices.min()
        max_px = prices.max()

        if min_px == max_px:
            return

        # Create histogram
        hist, bins = np.histogram(prices, bins=self.n_bins, range=(min_px, max_px), weights=volumes)

        # Center of bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        profile_df = pd.DataFrame({'price': bin_centers, 'volume': hist})
        profile_df = profile_df.sort_values('price')

        self.profiles[symbol] = profile_df

    def get_key_levels(self, symbol: str) -> Dict[str, float]:
        """Return POC, VAH, VAL."""
        if symbol not in self.profiles:
            return {}

        df = self.profiles[symbol]
        total_volume = df['volume'].sum()

        if total_volume == 0:
             return {}

        # 1. Point of Control (POC)
        max_idx = df['volume'].idxmax()
        poc = df.loc[max_idx, 'price']

        # 2. Value Area (VA)
        # Start at POC and expand out until we cover 70% volume
        sorted_by_vol = df.sort_values('volume', ascending=False)
        cum_vol = sorted_by_vol['volume'].cumsum()

        # Filter for rows in value area
        df_va = sorted_by_vol[cum_vol <= (total_volume * self.value_area_pct)]

        vah = df_va['price'].max()
        val = df_va['price'].min()

        return {
            "POC": float(poc),
            "VAH": float(vah),
            "VAL": float(val)
        }

    def get_nodes(self, symbol: str, threshold_pct: float = 0.8) -> Dict[str, List[float]]:
        """Return High and Low Volume Nodes."""
        if symbol not in self.profiles:
            return {}

        df = self.profiles[symbol]
        mean_vol = df['volume'].mean()

        # HVN: Local peaks significantly above mean
        # LVN: Local valleys significantly below mean

        # Simple thresholding
        hvns = df[df['volume'] > (mean_vol * (1 + threshold_pct))]['price'].tolist()
        lvns = df[df['volume'] < (mean_vol * (1 - threshold_pct))]['price'].tolist()

        # Clustering could be better, but this is MVP
        return {
            "HVN": hvns,
            "LVN": lvns
        }
