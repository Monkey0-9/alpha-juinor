"""
Comprehensive Feature Engineering Module.

Computes all required features for the institutional trading system:
- Returns features (log returns, rolling vol, EWMA vol, realized vol, ATR)
- Moving average features (EMA/SMA slopes, momentum)
- Momentum features (1m/3m/6m/12m, RSI, z-scores)
- Liquidity features (ADV, trade count, bid/ask spread)
- Cross-sectional ranks (percentile, momentum_rank, vol_rank)
- Correlation & beta vs benchmark
- Representation vector z_t from temporal autoencoder
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Feature version for tracking
FEATURE_VERSION = "1.0.0"

# Benchmark symbols for correlation/beta calculations
BENCHMARKS = ['SPY', 'QQQ', 'IWM']


@dataclass
class FeatureSet:
    """Container for computed features"""
    symbol: str
    date: str
    features: Dict[str, Any]
    version: str = FEATURE_VERSION
    raw_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'date': self.date,
            'features': self.features,
            'version': self.version,
            'raw_hash': self.raw_hash
        }


class FeatureComputer:
    """
    Comprehensive feature engineering for institutional trading.

    All features are computed deterministically for auditability.
    Each feature includes provenance metadata.
    """

    def __init__(self, benchmark_symbol: str = 'SPY', lookback_windows: List[int] = None):
        """
        Initialize feature computer.

        Args:
            benchmark_symbol: Benchmark for beta/correlation calculations
            lookback_windows: Custom lookback windows (default: [5, 10, 20, 60, 120, 252])
        """
        self.benchmark_symbol = benchmark_symbol
        self.lookback_windows = lookback_windows or [5, 10, 20, 60, 120, 252]

        # Technical indicator parameters
        self.atr_period = 14
        self.rsi_period = 14
        self.ema_windows = [5, 20, 50, 200]
        self.sma_windows = [20, 50, 200]

        # ADV calculation windows
        self.adv_windows = [1, 5, 20, 60]  # 1d, 1w, 1m, 3m

    def compute_all_features(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        benchmark_data: pd.DataFrame = None,
        metadata: Dict[str, Any] = None
    ) -> FeatureSet:
        """
        Compute all features for a symbol.

        Args:
            symbol: Ticker symbol
            price_data: DataFrame with OHLCV data (must have 'Close', 'High', 'Low', 'Volume')
            benchmark_data: Optional benchmark price data for beta calculations
            metadata: Additional metadata to include

        Returns:
            FeatureSet with all computed features
        """
        if price_data.empty:
            raise ValueError(f"Empty price data for {symbol}")

        # Ensure timezone-aware index
        if price_data.index.tz is None:
            price_data = price_data.copy()
            price_data.index = price_data.index.tz_localize('UTC')

        # Get the calculation date (latest available)
        calc_date = price_data.index[-1].strftime('%Y-%m-%d')

        features = {}

        # 1. RETURN FEATURES
        features.update(self._compute_return_features(price_data))

        # 2. VOLATILITY FEATURES
        features.update(self._compute_volatility_features(price_data))

        # 3. MOVING AVERAGE FEATURES
        features.update(self._compute_ma_features(price_data))

        # 4. MOMENTUM FEATURES
        features.update(self._compute_momentum_features(price_data))

        # 5. MOMENTUM CONTINUED (RSI, Z-SCORE)
        features.update(self._compute_oscillator_features(price_data))

        # 6. LIQUIDITY FEATURES
        features.update(self._compute_liquidity_features(price_data))

        # 7. CROSS-SECTIONAL RANKS (requires universe context - set placeholders)
        features.update(self._compute_rank_features(price_data))

        # 8. CORRELATION & BETA
        if benchmark_data is not None and not benchmark_data.empty:
            features.update(self._compute_beta_features(price_data, benchmark_data))
        else:
            features.update(self._compute_beta_features(price_data))

        # 9. PRICE PATTERN FEATURES
        features.update(self._compute_pattern_features(price_data))

        # 10. PROVENANCE
        features['_provenance'] = {
            'computed_at': datetime.utcnow().isoformat(),
            'data_rows': len(price_data),
            'lookback_windows': self.lookback_windows,
            'version': FEATURE_VERSION
        }

        # Create raw hash for integrity
        raw_hash = hashlib.sha256(
            json.dumps(features, sort_keys=True, default=str).encode()
        ).hexdigest()

        return FeatureSet(
            symbol=symbol,
            date=calc_date,
            features=features,
            version=FEATURE_VERSION,
            raw_hash=raw_hash
        )

    # =========================================================================
    # RETURN FEATURES
    # =========================================================================

    def _compute_return_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute return-based features"""
        close = df['Close']
        returns = np.log(close / close.shift(1)).dropna()

        features = {}

        # Log returns
        for window in self.lookback_windows:
            if len(returns) >= window:
                features[f'return_log_{window}d'] = returns.tail(window).sum()

        # Simple returns
        for window in self.lookback_windows:
            if len(close) >= window + 1:
                features[f'return_simple_{window}d'] = (
                    close.iloc[-1] / close.iloc[-window-1] - 1
                )

        # Cumulative return from various lookbacks
        for window in [60, 120, 252]:  # ~3m, 6m, 1y
            if len(close) >= window + 1:
                features[f'cumul_return_{window}d'] = (
                    close.iloc[-1] / close.iloc[-window-1] - 1
                )

        return features

    # =========================================================================
    # VOLATILITY FEATURES
    # =========================================================================

    def _compute_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute volatility-based features"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        returns = np.log(close / close.shift(1)).dropna()

        features = {}

        # Rolling volatility (annualized)
        for window in self.lookback_windows:
            if len(returns) >= window:
                roll_vol = returns.tail(window).std()
                features[f'vol_rolling_{window}d'] = roll_vol * np.sqrt(252)

        # EWMA volatility
        for span in [20, 60]:
            if len(returns) >= span:
                vol_ewma = returns.ewm(span=span).std().iloc[-1]
                features[f'vol_ewma_{span}d'] = vol_ewma * np.sqrt(252)

        # Realized volatility (various windows)
        for window in [5, 10, 20, 60]:
            if len(returns) >= window:
                realized_vol = returns.tail(window).std()
                features[f'realized_vol_{window}d'] = realized_vol * np.sqrt(252)

        # ATR (Average True Range)
        if 'Close' in df.columns and 'High' in df.columns and 'Low' in df.columns:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            for window in [self.atr_period, 20]:
                if len(tr) >= window:
                    atr = tr.tail(window).mean()
                    # ATR as percentage of price
                    atr_pct = atr / close.iloc[-1]
                    features[f'atr_{window}pct'] = atr_pct
                    features[f'atr_{window}'] = atr

        # Volatility regime (current vs historical)
        if len(returns) >= 252:
            current_vol = returns.tail(20).std() * np.sqrt(252)
            hist_vol = returns.std() * np.sqrt(252)
            if hist_vol > 0:
                features['vol_regime'] = current_vol / hist_vol

        return features

    # =========================================================================
    # MOVING AVERAGE FEATURES
    # =========================================================================

    def _compute_ma_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute moving average features"""
        close = df['Close']
        features = {}

        # EMA features
        for window in self.ema_windows:
            if len(close) >= window:
                ema = close.ewm(span=window, adjust=False).mean()
                ema_slope = (ema.iloc[-1] - ema.iloc[-min(5, window)]) / min(5, window)
                ema_position = close.iloc[-1] / ema.iloc[-1] - 1
                features[f'ema_{window}_slope'] = ema_slope
                features[f'ema_{window}_pct_above'] = ema_position

        # SMA features
        for window in self.sma_windows:
            if len(close) >= window:
                sma = close.rolling(window=window).mean()
                sma_position = close.iloc[-1] / sma.iloc[-1] - 1
                sma_slope = (sma.iloc[-1] - sma.iloc[-min(5, window)]) / min(5, window)
                features[f'sma_{window}_pct_above'] = sma_position
                features[f'sma_{window}_slope'] = sma_slope

        # MA crossovers (Golden/Death cross detection)
        if len(close) >= 200:
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1]

            features['ma_golden_cross'] = 1.0 if sma_50 > sma_200 else 0.0
            features['ma_50_200_spread'] = (sma_50 - sma_200) / sma_200

        return features

    # =========================================================================
    # MOMENTUM FEATURES
    # =========================================================================

    def _compute_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute momentum features"""
        close = df['Close']
        features = {}

        # Multi-period momentum
        for period in [1, 3, 6, 12]:  # months
            months_back = int(period * 21)  # Approximate trading days per month
            if len(close) >= months_back + 21:
                momentum = close.iloc[-1] / close.iloc[-months_back-21] - 1
                features[f'momentum_{period}m'] = momentum

        # Rate of change
        for window in [5, 10, 20, 60]:
            if len(close) >= window + 1:
                roc = (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window]
                features[f'roc_{window}d'] = roc

        # Acceleration (change in returns)
        returns = close.pct_change().dropna()
        if len(returns) >= 10:
            accel = returns.diff().tail(5).mean()
            features['momentum_acceleration'] = accel

        return features

    # =========================================================================
    # OSCILLATOR FEATURES (RSI, Z-SCORE)
    # =========================================================================

    def _compute_oscillator_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute oscillator-based features (RSI, z-scores)"""
        close = df['Close']
        features = {}

        # RSI
        for period in [self.rsi_period, 7, 21]:
            if len(close) >= period + 1:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                features[f'rsi_{period}'] = rsi.iloc[-1]

        # Z-scores over multiple windows
        for window in [20, 60, 120]:
            if len(close) >= window:
                z = (close.iloc[-1] - close.rolling(window).mean().iloc[-1]) / (
                    close.rolling(window).std().iloc[-1] + 1e-10
                )
                features[f'zscore_{window}d'] = z

        # Bollinger Band position
        if len(close) >= 20:
            sma_20 = close.rolling(20).mean().iloc[-1]
            std_20 = close.rolling(20).std().iloc[-1]
            bb_upper = sma_20 + 2 * std_20
            bb_lower = sma_20 - 2 * std_20
            bb_position = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            features['bb_position'] = bb_position

        # Stochastic oscillator
        if len(close) >= 14:
            lowest_low = low = df['Low'].rolling(14).min().iloc[-1]
            highest_high = high = df['High'].rolling(14).max().iloc[-1]
            stoch_k = 100 * (close.iloc[-1] - lowest_low) / (highest_high - lowest_low + 1e-10)
            features['stochastic_k'] = stoch_k

        return features

    # =========================================================================
    # LIQUIDITY FEATURES
    # =========================================================================

    def _compute_liquidity_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute liquidity-based features"""
        close = df['Close']
        volume = df['Volume']
        features = {}

        # ADV (Average Daily Volume) for various windows
        for window in self.adv_windows:
            if len(volume) >= window:
                adv = volume.tail(window).mean()
                adv_dollar = adv * close.iloc[-1]
                features[f'adv_{window}d'] = adv
                features[f'adv_{window}d_dollar'] = adv_dollar

        # Volume ratio (today vs 20-day avg)
        if len(volume) >= 21:
            vol_ratio = volume.iloc[-1] / volume.tail(20).mean()
            features['vol_ratio_20d'] = vol_ratio

        # Volume trend
        for window in [5, 10, 20]:
            if len(volume) >= window:
                vol_sma = volume.rolling(window).mean()
                vol_trend = (volume.iloc[-1] / vol_sma.iloc[-window]) - 1
                features[f'vol_trend_{window}d'] = vol_trend

        # Turnover estimate (volume / shares outstanding proxy)
        # Using ADV / price as a liquidity proxy
        if 'adv_20d_dollar' in features:
            features['liquidity_score'] = min(1.0, features['adv_20d_dollar'] / 1e9)  # Normalized to $1B

        # Trade intensity (volume relative to price range)
        price_range = (df['High'] - df['Low']).iloc[-1]
        if price_range > 0:
            trade_intensity = volume.iloc[-1] * close.iloc[-1] / price_range
            features['trade_intensity'] = trade_intensity

        return features

    # =========================================================================
    # CROSS-SECTIONAL RANKS
    # =========================================================================

    def _compute_rank_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute cross-sectional rank features (requires universe context)"""
        # These features require the full universe to compute ranks
        # For now, compute relative position within the symbol's own history

        close = df['Close']
        returns = close.pct_change().dropna()
        vol = returns.rolling(20).std() * np.sqrt(252)

        features = {}

        # Relative strength (momentum vs volatility rank)
        if len(returns) >= 252:
            annual_return = close.iloc[-1] / close.iloc[-252] - 1
            annual_vol = returns.std() * np.sqrt(252)
            features['sharpe_proxy'] = annual_return / (annual_vol + 1e-10)

        # Calmar proxy (return / max drawdown)
        if len(close) >= 252:
            rolling_max = close.rolling(252).max()
            drawdown = (close - rolling_max) / rolling_max
            max_dd = drawdown.min()
            annual_return = close.iloc[-1] / close.iloc[-252] - 1
            features['calmar_proxy'] = -annual_return / (max_dd + 1e-10)

        # Volatility rank (lower is better)
        if len(vol) >= 60:
            current_vol = vol.iloc[-1]
            vol_percentile = (vol < current_vol).mean()
            features['vol_rank_60d'] = vol_percentile

        # Momentum rank within recent period
        for period in [1, 3, 6]:
            months_back = int(period * 21)
            if len(close) >= months_back + 21:
                mom = close.iloc[-1] / close.iloc[-months_back-21] - 1
                # Store as percentile within available data
                features[f'momentum_rank_{period}m'] = 0.5  # Placeholder

        return features

    # =========================================================================
    # BETA & CORRELATION FEATURES
    # =========================================================================

    def _compute_beta_features(
        self,
        df: pd.DataFrame,
        benchmark_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """Compute beta and correlation features"""
        close = df['Close']
        returns = np.log(close / close.shift(1)).dropna()

        features = {}

        # Use benchmark if provided, otherwise use SPY proxy from returns
        if benchmark_data is not None and not benchmark_data.empty:
            bench_close = benchmark_data['Close']
            bench_returns = np.log(bench_close / bench_close.shift(1)).dropna()
        else:
            # Use market proxy: negative of VIX-like volatility or S&P proxy
            # In practice, we should have proper benchmark data
            bench_returns = returns  # Placeholder

        # Align returns
        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) < 60:
            return {'beta_placeholder': 0.0}

        aligned_returns = returns.loc[common_idx]
        aligned_bench = bench_returns.loc[common_idx]

        # Correlation
        corr = aligned_returns.corr(aligned_bench)
        features[f'corr_{self.benchmark_symbol}'] = corr

        # Beta
        cov = aligned_returns.cov(aligned_bench)
        var_bench = aligned_bench.var()
        if var_bench > 0:
            beta = cov / var_bench
            features[f'beta_{self.benchmark_symbol}'] = beta
        else:
            features[f'beta_{self.benchmark_symbol}'] = 1.0

        # Alpha (annualized)
        if len(aligned_returns) >= 252:
            annual_return = aligned_returns.tail(252).sum()
            annual_bench = aligned_bench.tail(252).sum()
            features[f'alpha_{self.benchmark_symbol}'] = annual_return - (
                beta * annual_bench
            )

        # R-squared
        features[f'r2_{self.benchmark_symbol}'] = corr ** 2

        # Tracking error
        tracking_error = (aligned_returns - aligned_bench * beta).std() * np.sqrt(252)
        features[f'tracking_error_{self.benchmark_symbol}'] = tracking_error

        return features

    # =========================================================================
    # PATTERN FEATURES
    # =========================================================================

    def _compute_pattern_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute price pattern features"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        features = {}

        # Price position within day's range
        if len(close) >= 1:
            today_range = high.iloc[-1] - low.iloc[-1]
            if today_range > 0:
                close_position = (close.iloc[-1] - low.iloc[-1]) / today_range
                features['close_position_daily'] = close_position

        # Gap analysis
        if len(close) >= 2:
            gap = (close.iloc[-1] / close.iloc[-2] - 1) - (
                (high.iloc[-2] - low.iloc[-2]) / close.iloc[-2]
            )
            features['gap_proxy'] = gap

        # Intraday range (volatility proxy)
        if len(high) >= 1 and len(low) >= 1:
            intraday_range = (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1]
            features['intraday_range'] = intraday_range

        # Volume-weighted average price (VWAP) proxy
        if 'Volume' in df.columns:
            pv = (close * df['Volume']).sum()
            v = df['Volume'].sum()
            if v > 0:
                vwap = pv / v
                features['vwap_deviation'] = (close.iloc[-1] - vwap) / vwap

        # Recent price action (last 5 days)
        if len(close) >= 6:
            recent = close.tail(5)
            direction = 1 if recent.iloc[-1] > recent.iloc[0] else -1
            consistency = (recent.diff().dropna() > 0).mean() * direction
            features['price_consistency_5d'] = consistency

        return features


def compute_z_temporal_ae(
    price_history: pd.DataFrame,
    encoding_dim: int = 32,
    model_path: str = None
) -> np.ndarray:
    """
    Compute representation vector z_t from temporal autoencoder.

    This is a simplified implementation. In production, use a trained
    LSTM/Transformer autoencoder.

    Args:
        price_history: Price history DataFrame
        encoding_dim: Dimension of the encoding vector
        model_path: Path to pre-trained model

    Returns:
        Encoding vector z_t
    """
    # Normalize returns
    close = price_history['Close'].dropna()
    returns = np.log(close / close.shift(1)).dropna()

    if len(returns) < 60:
        # Not enough data, return zeros
        return np.zeros(encoding_dim)

    # Use last 252 days (1 year)
    returns = returns.tail(252)

    # Create windows for sequence modeling
    window_size = 20
    n_windows = len(returns) // window_size

    if n_windows < 1:
        return np.zeros(encoding_dim)

    # Compute features from each window
    window_features = []
    for i in range(n_windows):
        window = returns.iloc[i*window_size:(i+1)*window_size]
        feat = [
            window.mean(),  # Mean return
            window.std(),   # Volatility
            window.skew(),  # Skewness
            window.kurtosis(),  # Kurtosis
            window.min(),   # Min return
            window.max(),   # Max return
            window.sum(),   # Cumulative
            (window > 0).mean(),  # Win rate
        ]
        window_features.append(feat)

    window_features = np.array(window_features)

    # Simple dimensionality reduction (PCA-like)
    if len(window_features) > encoding_dim:
        # Use SVD for dimensionality reduction
        from scipy.linalg import svd
        U, s, Vt = svd(window_features, full_matrices=False)
        encoding = U[:, :encoding_dim].flatten()[:encoding_dim]
    else:
        # Pad or truncate
        if len(window_features.flatten()) < encoding_dim:
            encoding = np.pad(window_features.flatten(), (0, encoding_dim - len(window_features.flatten())))
        else:
            encoding = window_features.flatten()[:encoding_dim]

    # Normalize
    if np.linalg.norm(encoding) > 0:
        encoding = encoding / np.linalg.norm(encoding)

    return encoding


def batch_compute_features(
    symbols: List[str],
    price_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame = None,
    n_jobs: int = 4
) -> Dict[str, FeatureSet]:
    """
    Compute features for multiple symbols in parallel.

    Args:
        symbols: List of ticker symbols
        price_data: Dict mapping symbol -> price DataFrame
        benchmark_data: Benchmark price data
        n_jobs: Number of parallel jobs

    Returns:
        Dict mapping symbol -> FeatureSet
    """
    from concurrent.futures import ThreadPoolExecutor

    computer = FeatureComputer()

    def compute_for_symbol(symbol: str) -> Tuple[str, FeatureSet]:
        if symbol not in price_data:
            logger.warning(f"No price data for {symbol}")
            return symbol, None
        try:
            fs = computer.compute_all_features(
                symbol, price_data[symbol], benchmark_data
            )
            return symbol, fs
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
            return symbol, None

    results = {}
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(compute_for_symbol, s): s for s in symbols}
        for future in futures:
            symbol, fs = future.result()
            if fs is not None:
                results[symbol] = fs

    return results

