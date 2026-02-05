"""
Alpha Factors Module
====================

This module defines the core quantitative factors used for signal generation.
It implements a vectorized "Alpha Factory" architecture.

Core Components:
- Factor: Abstract base class for all alpha factors.
- VolScaledMomentum: Time-series momentum scaled by realized volatility.
"""

import abc
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class Factor(abc.ABC):
    """Abstract base class for all alpha factors."""

    @abc.abstractmethod
    def compute(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute factor scores for the given market data.

        Args:
            market_data: DataFrame with MultiIndex (Ticker, Date) or similar structure,
                         containing at least 'Close' prices.

        Returns:
            DataFrame of factor scores (Tickers x Factors or Single Series).
        """
        pass

class VolScaledMomentum(Factor):
    """
    Volatility-Scaled Momentum Factor.

    Formula:
        Mom_score = (Price_t / Price_{t-K} - 1) / (Volatility_t)

    Where:
        - K is the lookback window (e.g., 126 days for 6-month momentum).
        - Volatility_t is the realized volatility over a shorter window (e.g., 20 days).

    The raw scores are then standardized cross-sectionally (Z-Score) to ensure
    comparability across assets and time.
    """

    def __init__(self, momentum_window: int = 126, vol_window: int = 20):
        self.momentum_window = momentum_window
        self.vol_window = vol_window

    def compute(self, market_data: pd.DataFrame) -> pd.Series:
        """
        Compute annualized, vol-scaled momentum scores.
        """
        # Ensure we have a DataFrame of Close prices
        if isinstance(market_data.columns, pd.MultiIndex):
            # Extract 'Close' if it's a multi-index (Ticker, Field) or (Field, Ticker) construction
            # Strategy passes data usually as columns=Tickers or similar.
            # Let's assume input is a DataFrame where columns are Tickers and values are Prices
            # OR a standard OHLCV frame.

            # Case A: Standard OHLCV with MultiIndex columns (Ticker -> OHLCV)
            # We need to construct a "Close" dataframe
            closes = pd.DataFrame()
            tickers = market_data.columns.get_level_values(0).unique()
            for ticker in tickers:
                if 'Close' in market_data[ticker].columns:
                    closes[ticker] = market_data[ticker]['Close']

        else:
            # Case B: columns are Tickers, values are Close prices (simplified input)
            # OR Case C: Standard OHLCV single ticker (not applicable for cross-section really)
            # We will assume Case B if it looks like prices, otherwise try to find 'Close'
            if 'Close' in market_data.columns:
                 # Single ticker passed as frame
                 # This is edge case. We want Muti-Ticker.
                 closes = market_data[['Close']]
            else:
                 closes = market_data

        if closes.empty:
            logger.warning("VolScaledMomentum: No close prices found.")
            return pd.Series()

        # 1. Calculate Returns (Momentum)
        # We use log returns for better additivity over long horizons, or simple returns.
        # Simple formula: P_t / P_{t-K} - 1
        momentum = closes.pct_change(periods=self.momentum_window)

        # 2. Calculate Volatility (Realized)
        # Standard deviation of daily returns * sqrt(252)
        daily_rets = closes.pct_change(1)
        volatility = daily_rets.rolling(window=self.vol_window).std() * np.sqrt(252)

        # 3. Vol-Scaled Momentum
        # Avoid division by zero
        vol_scaled_mom = momentum / volatility.replace(0, np.nan)

        # 4. Get the latest slice (cross-section at time t)
        # We generally want the most recent valid signal
        latest_scores = vol_scaled_mom.iloc[-1]

        # 5. Cross-Sectional Ranking (Z-Score)
        # (Score - Mean) / Std
        # Only rank if we have enough assets (>2)
        if len(latest_scores.dropna()) > 2:
            mu = latest_scores.mean()
            sigma = latest_scores.std()
            if sigma > 0:
                standardized_scores = (latest_scores - mu) / sigma
            else:
                standardized_scores = latest_scores - mu # Flat
        else:
            standardized_scores = latest_scores

        return standardized_scores.fillna(0.0)

class MeanReversion(Factor):
    """
    Bollinger Band Mean Reversion Factor.

    Score is based on distance from the moving average relative to volatility.
    High Score = Price is Low (Oversold) -> Buy Signal.
    Low Score = Price is High (Overbought) -> Sell Signal.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, market_data: pd.DataFrame) -> pd.Series:
        # Extract closes similar to above
        if isinstance(market_data.columns, pd.MultiIndex):
            closes = pd.DataFrame()
            tickers = market_data.columns.get_level_values(0).unique()
            for ticker in tickers:
                if 'Close' in market_data[ticker].columns:
                    closes[ticker] = market_data[ticker]['Close']
        elif 'Close' in market_data.columns:
             closes = market_data[['Close']]
        else:
             closes = market_data

        if closes.empty:
            return pd.Series()

        # Calculate Z-Score of price relative to MA
        # z = (Price - MA) / StdDev
        ma = closes.rolling(window=self.window).mean()
        std = closes.rolling(window=self.window).std()

        z_score = (closes - ma) / std.replace(0, np.nan)

        # Mean Reversion Logic:
        # We want to buy when Z is low (Oversold) and sell when Z is high (Overbought).
        # So Alpha Score should be inverse of Z-Score.
        # If Z = -2 (Oversold), Score = +2 (Buy).

        raw_scores = -1.0 * z_score.iloc[-1]

        return raw_scores.fillna(0.0)
