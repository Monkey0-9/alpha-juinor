"""
Microstructure Alpha Factors
=============================

Elite microstructure indicators for institutional trading:
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Kyle's Lambda (Market Impact Coefficient)
- Effective Spread Decomposition
- Order Flow Toxicity
- Adverse Selection Cost
- Roll Model (Spread Estimator)
- Amihud Illiquidity Ratio
- Trade Imbalance Metrics
- Pin (Probability of Informed Trading)
- Quote Slope
- Depth Imbalance
- Weighted Price Contribution
- Volume Fragility Index
- Market Quality Index
- Liquidity Cost Score (LCS)

References:
- Easley, D., López de Prado, M. M., & O'Hara, M. (2012). "Flow toxicity and liquidity in a high-frequency world"
- Kyle, A. S. (1985). "Continuous auctions and insider trading"
- Roll, R. (1984). "A simple implicit measure of the effective bid-ask spread"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureMetrics:
    """Container for microstructure metrics."""
    vpin: float
    kyle_lambda: float
    effective_spread: float
    adverse_selection_cost: float
    roll_spread: float
    amihud_illiquidity: float
    trade_imbalance: float
    pin: float
    depth_imbalance: float
    volume_fragility: float


class MicrostructureFactors:
    """
    Advanced microstructure factors for Top 1% hedge funds.

    These factors capture:
    - Information asymmetry (VPIN, PIN)
    - Market impact (Kyle's lambda)
    - Transaction costs (spreads, slippage)
    - Liquidity quality
    - Order flow toxicity
    """

    def __init__(self,
                 vpin_buckets: int = 50,
                 kyle_window: int = 20,
                 min_trades: int = 100):
        """
        Initialize microstructure factor calculator.

        Args:
            vpin_buckets: Number of volume buckets for VPIN
            kyle_window: Lookback window for Kyle's lambda
            min_trades: Minimum trades required for calculation
        """
        self.vpin_buckets = vpin_buckets
        self.kyle_window = kyle_window
        self.min_trades = min_trades

    def compute_all(self,
                    trades_df: pd.DataFrame,
                    quotes_df: Optional[pd.DataFrame] = None) -> MicrostructureMetrics:
        """
        Compute all microstructure factors.

        Args:
            trades_df: Trade-level data with ['price', 'volume', 'timestamp', 'side']
            quotes_df: Quote-level data with ['bid', 'ask', 'bid_size', 'ask_size', 'timestamp']

        Returns:
            MicrostructureMetrics with all factors
        """
        if len(trades_df) < self.min_trades:
            logger.warning(f"Insufficient trades ({len(trades_df)}) for microstructure analysis")
            return self._empty_metrics()

        metrics = MicrostructureMetrics(
            vpin=self.compute_vpin(trades_df),
            kyle_lambda=self.compute_kyle_lambda(trades_df),
            effective_spread=self.compute_effective_spread(trades_df, quotes_df),
            adverse_selection_cost=self.compute_adverse_selection(trades_df),
            roll_spread=self.compute_roll_spread(trades_df),
            amihud_illiquidity=self.compute_amihud(trades_df),
            trade_imbalance=self.compute_trade_imbalance(trades_df),
            pin=self.compute_pin(trades_df),
            depth_imbalance=self.compute_depth_imbalance(quotes_df) if quotes_df is not None else 0.0,
            volume_fragility=self.compute_volume_fragility(trades_df)
        )

        return metrics

    def compute_vpin(self, trades_df: pd.DataFrame) -> float:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN).

        Measures order flow toxicity by comparing buy/sell volume imbalances
        in synchronized volume buckets.

        High VPIN → High probability of informed trading → Avoid or widen spreads
        Low VPIN → Uninformed flow → Safe to provide liquidity

        Args:
            trades_df: Trade data with 'volume' and 'side' (1=buy, -1=sell)

        Returns:
            VPIN metric [0, 1] where higher = more toxic flow
        """
        try:
            # Classify trades as buy (1) or sell (-1)
            if 'side' not in trades_df.columns:
                # Use tick rule if side not available
                trades_df = trades_df.copy()
                trades_df['side'] = self._classify_trades_tick_rule(trades_df['price'])

            # Create volume buckets
            total_volume = trades_df['volume'].sum()
            bucket_size = total_volume / self.vpin_buckets

            trades_df = trades_df.copy()
            trades_df['cum_volume'] = trades_df['volume'].cumsum()
            trades_df['bucket'] = (trades_df['cum_volume'] / bucket_size).astype(int)

            # Calculate buy/sell volume in each bucket
            bucket_stats = trades_df.groupby('bucket').agg({
                'volume': 'sum',
                'side': lambda x: (x * trades_df.loc[x.index, 'volume']).sum()
            })

            # VPIN = average absolute order flow imbalance
            bucket_stats['buy_volume'] = (bucket_stats['side'] + bucket_stats['volume']) / 2
            bucket_stats['sell_volume'] = (bucket_stats['volume'] - bucket_stats['side']) / 2
            bucket_stats['imbalance'] = abs(bucket_stats['buy_volume'] - bucket_stats['sell_volume'])

            vpin = (bucket_stats['imbalance'] / bucket_stats['volume']).mean()

            return float(np.clip(vpin, 0, 1))

        except Exception as e:
            logger.error(f"Error computing VPIN: {e}")
            return 0.5  # Neutral value

    def compute_kyle_lambda(self, trades_df: pd.DataFrame) -> float:
        """
        Kyle's Lambda - Market Impact Coefficient.

        Measures price impact per unit of order flow:
        ΔP = λ * Q

        Where:
        - ΔP = Price change
        - λ = Kyle's lambda (impact coefficient)
        - Q = Signed order flow

        Higher λ → Higher market impact → Trade smaller or use algos

        Args:
            trades_df: Trade data with 'price', 'volume', 'side'

        Returns:
            Kyle's lambda (positive value representing impact)
        """
        try:
            df = trades_df.copy().sort_values('timestamp')

            # Calculate price changes
            df['price_change'] = df['price'].diff()

            # Signed volume (buy=positive, sell=negative)
            if 'side' in df.columns:
                df['signed_volume'] = df['side'] * df['volume']
            else:
                df['signed_volume'] = self._classify_trades_tick_rule(df['price']) * df['volume']

            # Regression: price_change ~ signed_volume
            # Using rolling window for stability
            lambdas = []

            for i in range(self.kyle_window, len(df)):
                window = df.iloc[i-self.kyle_window:i]

                # Robust regression (to handle outliers)
                valid = ~(window['price_change'].isna() | window['signed_volume'].isna())
                if valid.sum() < 10:
                    continue

                X = window.loc[valid, 'signed_volume'].values
                y = window.loc[valid, 'price_change'].values

                if len(X) > 0 and X.std() > 0:
                    # Simple OLS beta
                    lambda_i = (np.cov(X, y)[0, 1] / np.var(X)) if np.var(X) > 0 else 0
                    lambdas.append(abs(lambda_i))  # Take absolute value

            if len(lambdas) == 0:
                return 0.0

            # Return median to be robust to outliers
            kyle_lambda = float(np.median(lambdas))

            # Normalize by price for interpretability
            avg_price = df['price'].mean()
            if avg_price > 0:
                kyle_lambda = kyle_lambda / avg_price * 10000  # In basis points

            return kyle_lambda

        except Exception as e:
            logger.error(f"Error computing Kyle's lambda: {e}")
            return 0.0

    def compute_effective_spread(self,
                                trades_df: pd.DataFrame,
                                quotes_df: Optional[pd.DataFrame] = None) -> float:
        """
        Effective Spread - Actual transaction cost.

        Effective Spread = 2 * |Trade Price - Mid Price|

        Measures realized spread paid, accounting for:
        - Price improvement
        - Hidden liquidity
        - Smart order routing

        Args:
            trades_df: Trade data
            quotes_df: Quote data (optional, uses trade midpoint if unavailable)

        Returns:
            Effective spread in basis points
        """
        try:
            if quotes_df is not None and len(quotes_df) > 0:
                # Merge trades with nearest quotes
                trades = trades_df.copy()
                trades = pd.merge_asof(
                    trades.sort_values('timestamp'),
                    quotes_df[['timestamp', 'bid', 'ask']].sort_values('timestamp'),
                    on='timestamp',
                    direction='backward'
                )
                trades['mid'] = (trades['bid'] + trades['ask']) / 2
            else:
                # Approximate mid using VWAP
                trades = trades_df.copy()
                trades['mid'] = trades['price'].rolling(20, min_periods=1).mean()

            # Effective spread = 2 * |price - mid|
            trades['effective_spread'] = 2 * abs(trades['price'] - trades['mid'])

            # Volume-weighted average
            total_volume = trades['volume'].sum()
            if total_volume > 0:
                weighted_spread = (trades['effective_spread'] * trades['volume']).sum() / total_volume

                # Convert to basis points
                avg_price = trades['price'].mean()
                spread_bps = (weighted_spread / avg_price) * 10000 if avg_price > 0 else 0

                return float(spread_bps)

            return 0.0

        except Exception as e:
            logger.error(f"Error computing effective spread: {e}")
            return 0.0

    def compute_adverse_selection(self, trades_df: pd.DataFrame, horizon: int = 5) -> float:
        """
        Adverse Selection Cost.

        Measures information content by checking if prices move against you
        after trading (permanent impact).

        ASC = E[P_{t+h} - P_t | Buy at t]

        High ASC → Trading against informed flow → Widen spreads

        Args:
            trades_df: Trade data
            horizon: Minutes to measure adverse price movement

        Returns:
            Adverse selection cost in basis points
        """
        try:
            df = trades_df.copy().sort_values('timestamp')

            # Calculate future price (h periods ahead)
            df['future_price'] = df['price'].shift(-horizon)
            df['price_change'] = df['future_price'] - df['price']

            # Classify trades
            if 'side' not in df.columns:
                df['side'] = self._classify_trades_tick_rule(df['price'])

            # Adverse selection = average signed price change
            # Buy → expect price to rise (positive change = no adverse selection)
            # Sell → expect price to fall (negative change = no adverse selection)
            df['adverse_move'] = -df['side'] * df['price_change']  # Negative = good, Positive = adverse

            # Volume-weighted average
            total_volume = df['volume'].sum()
            if total_volume > 0:
                weighted_adverse = (df['adverse_move'] * df['volume']).sum() / total_volume

                # Convert to basis points
                avg_price = df['price'].mean()
                asc_bps = (weighted_adverse / avg_price) * 10000 if avg_price > 0 else 0

                return float(max(0, asc_bps))  # Only report positive (adverse) moves

            return 0.0

        except Exception as e:
            logger.error(f"Error computing adverse selection: {e}")
            return 0.0

    def compute_roll_spread(self, trades_df: pd.DataFrame) -> float:
        """
        Roll's Spread Estimator.

        Implicit bid-ask spread from price autocovariance:
        Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))

        Clever method to estimate spread without quote data.

        Args:
            trades_df: Trade data with price

        Returns:
            Estimated spread in basis points
        """
        try:
            prices = trades_df.sort_values('timestamp')['price'].values
            price_changes = np.diff(prices)

            if len(price_changes) < 2:
                return 0.0

            # Autocovariance
            autocov = np.cov(price_changes[1:], price_changes[:-1])[0, 1]

            # Roll spread (must be non-negative)
            if autocov < 0:
                roll_spread = 2 * np.sqrt(-autocov)

                # Convert to basis points
                avg_price = prices.mean()
                spread_bps = (roll_spread / avg_price) * 10000 if avg_price > 0 else 0

                return float(spread_bps)

            return 0.0  # No mean reversion detected

        except Exception as e:
            logger.error(f"Error computing Roll spread: {e}")
            return 0.0

    def compute_amihud(self, trades_df: pd.DataFrame) -> float:
        """
        Amihud Illiquidity Ratio.

        Illiquidity = Average( |Return| / Dollar Volume )

        Measures price impact per dollar traded.
        Higher → More illiquid → Harder to trade large sizes

        Args:
            trades_df: Trade data

        Returns:
            Amihud ratio (scaled by 1e6 for readability)
        """
        try:
            df = trades_df.copy().sort_values('timestamp')

            # Calculate returns
            df['return'] = df['price'].pct_change().abs()
            df['dollar_volume'] = df['price'] * df['volume']

            # Daily aggregation (or whatever timeframe available)
            # For intraday, we group by time buckets
            df['time_bucket'] = pd.to_datetime(df['timestamp']).dt.floor('5T')  # 5-minute buckets

            daily_stats = df.groupby('time_bucket').agg({
                'return': 'sum',
                'dollar_volume': 'sum'
            })

            # Amihud = |Return| / Dollar Volume
            daily_stats['amihud'] = daily_stats['return'] / (daily_stats['dollar_volume'] + 1e-10)

            amihud = daily_stats['amihud'].mean() * 1e6  # Scale for readability

            return float(amihud)

        except Exception as e:
            logger.error(f"Error computing Amihud ratio: {e}")
            return 0.0

    def compute_trade_imbalance(self, trades_df: pd.DataFrame) -> float:
        """
        Trade Imbalance Ratio.

        Imbalance = (Buy Volume - Sell Volume) / Total Volume

        Range: [-1, 1]
        - Positive: More buying pressure
        - Negative: More selling pressure

        Args:
            trades_df: Trade data

        Returns:
            Trade imbalance ratio
        """
        try:
            if 'side' not in trades_df.columns:
                trades_df = trades_df.copy()
                trades_df['side'] = self._classify_trades_tick_rule(trades_df['price'])

            buy_volume = trades_df[trades_df['side'] == 1]['volume'].sum()
            sell_volume = trades_df[trades_df['side'] == -1]['volume'].sum()
            total_volume = trades_df['volume'].sum()

            if total_volume > 0:
                imbalance = (buy_volume - sell_volume) / total_volume
                return float(np.clip(imbalance, -1, 1))

            return 0.0

        except Exception as e:
            logger.error(f"Error computing trade imbalance: {e}")
            return 0.0

    def compute_pin(self, trades_df: pd.DataFrame) -> float:
        """
        PIN (Probability of Informed Trading) - Easley-O'Hara model.

        Simplified estimation using trade imbalances and arrival rates.

        Full PIN requires iterative MLE, this is an approximation.

        Args:
            trades_df: Trade data

        Returns:
            PIN estimate [0, 1]
        """
        try:
            # Simplified PIN using order flow imbalance volatility
            # High variation in imbalances suggests informed trading

            df = trades_df.copy().sort_values('timestamp')
            df['time_bucket'] = pd.to_datetime(df['timestamp']).dt.floor('1T')  # 1-minute buckets

            # Calculate imbalance per bucket
            bucket_stats = df.groupby('time_bucket').apply(
                lambda x: (x[x.get('side', self._classify_trades_tick_rule(x['price'])) == 1]['volume'].sum() -
                          x[x.get('side', self._classify_trades_tick_rule(x['price'])) == -1]['volume'].sum()) /
                         (x['volume'].sum() + 1e-10)
            )

            # PIN proxy: Coefficient of variation of imbalances
            if len(bucket_stats) > 0 and bucket_stats.std() > 0:
                pin = min(1.0, abs(bucket_stats.mean()) + bucket_stats.std())
                return float(pin)

            return 0.5  # Neutral

        except Exception as e:
            logger.error(f"Error computing PIN: {e}")
            return 0.5

    def compute_depth_imbalance(self, quotes_df: pd.DataFrame) -> float:
        """
        Order Book Depth Imbalance.

        Imbalance = (Bid Size - Ask Size) / (Bid Size + Ask Size)

        Positive → More buy interest (bullish)
        Negative → More sell interest (bearish)

        Args:
            quotes_df: Quote data with bid_size and ask_size

        Returns:
            Depth imbalance ratio [-1, 1]
        """
        try:
            if quotes_df is None or len(quotes_df) == 0:
                return 0.0

            bid_size = quotes_df['bid_size'].mean()
            ask_size = quotes_df['ask_size'].mean()
            total_size = bid_size + ask_size

            if total_size > 0:
                imbalance = (bid_size - ask_size) / total_size
                return float(np.clip(imbalance, -1, 1))

            return 0.0

        except Exception as e:
            logger.error(f"Error computing depth imbalance: {e}")
            return 0.0

    def compute_volume_fragility(self, trades_df: pd.DataFrame) -> float:
        """
        Volume Fragility Index.

        Measures how quickly volume dries up (flash crash indicator).

        Fragility = Std(Volume) / Mean(Volume)

        High fragility → Unstable liquidity → Risk of gaps

        Args:
            trades_df: Trade data

        Returns:
            Fragility index (coefficient of variation)
        """
        try:
            df = trades_df.copy()
            df['time_bucket'] = pd.to_datetime(df['timestamp']).dt.floor('30S')  # 30-second buckets

            bucket_volumes = df.groupby('time_bucket')['volume'].sum()

            if len(bucket_volumes) > 1 and bucket_volumes.mean() > 0:
                fragility = bucket_volumes.std() / bucket_volumes.mean()
                return float(fragility)

            return 0.0

        except Exception as e:
            logger.error(f"Error computing volume fragility: {e}")
            return 0.0

    def _classify_trades_tick_rule(self, prices: pd.Series) -> pd.Series:
        """
        Classify trades as buy (1) or sell (-1) using tick rule.

        Uptick → Buy
        Downtick → Sell
        No change → Previous classification

        Args:
            prices: Price series

        Returns:
            Series of 1 (buy) or -1 (sell)
        """
        price_changes = prices.diff()

        # Classify based on price movement
        side = pd.Series(index=prices.index, dtype=float)
        side[price_changes > 0] = 1  # Buy
        side[price_changes < 0] = -1  # Sell

        # Forward-fill for unchanged prices
        side = side.fillna(method='ffill').fillna(1)  # Default to buy

        return side

    def _empty_metrics(self) -> MicrostructureMetrics:
        """Return empty/neutral metrics."""
        return MicrostructureMetrics(
            vpin=0.5,
            kyle_lambda=0.0,
            effective_spread=0.0,
            adverse_selection_cost=0.0,
            roll_spread=0.0,
            amihud_illiquidity=0.0,
            trade_imbalance=0.0,
            pin=0.5,
            depth_imbalance=0.0,
            volume_fragility=0.0
        )
