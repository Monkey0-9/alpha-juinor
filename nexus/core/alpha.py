"""
nexus/core/alpha.py — Superhuman Alpha Signal Engine

Replaces naive Kalman + Monte Carlo blend with:
  - Entropy-filtered momentum (Shannon entropy down-weights noisy signals)
  - Hurst-gated signals (H>0.55 → momentum, H<0.45 → mean-reversion)
  - VWAP deviation alpha (smart money tracking)
  - Adaptive Kalman variance (process/measurement updated from rolling vol)
  - Regime-adaptive signal weighting
  - Multi-factor signal combination with IC-optimal blending
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Tuple
from nexus.math.models import KalmanFilter
from nexus.math.indicators import HawkesProcess, compute_hurst_exponent
from nexus.execution.alpaca import get_client
from nexus.utils.config import Config
from nexus.core.sentiment import SentimentEngine

logger = logging.getLogger(__name__)


class AdaptiveKalmanFilter(KalmanFilter):
    """
    Adaptive Kalman Filter that updates its process/measurement variances
    dynamically from rolling observed volatility.

    Tight variance (low vol) → more trust in model prediction
    Wide variance (high vol) → more trust in new measurement
    """

    def adapt_to_volatility(self, rolling_vol: float) -> None:
        """
        Update Q (process variance) and R (measurement variance)
        based on observed market volatility.

        High vol → increase R (measurement noise) → smoother filter
        Low vol  → decrease R → faster tracking
        """
        # Process variance (how much 'true' price drifts per step)
        self.process_variance = max(1e-7, rolling_vol**2 * 0.1)
        # Measurement variance (sensor noise / microstructure noise)
        self.measurement_variance = max(1e-5, rolling_vol**2 * 5.0)


class AlphaEngine:
    """
    Superhuman Alpha Generation Engine.

    Multi-factor signal combining:
    1. Adaptive Kalman-denoised trend
    2. Entropy-filtered momentum (suppresses noisy distributions)
    3. Hurst-gated directional signal (momentum vs. mean-reversion gate)
    4. VWAP deviation signal (smart money tracking)
    5. Hawkes Process intensity adjustment (vol clustering penalty)
    6. Monte Carlo probability adjustment
    """

    _cache: Dict[str, Dict[str, Any]] = {}
    _CACHE_TTL = 300  # 5 minutes

    def __init__(self, backend_url: str = Config.BACKEND_URL):
        self.kf = AdaptiveKalmanFilter()
        self.hawkes = HawkesProcess()
        self.backend_url = backend_url
        self.client = get_client()
        self.sentiment_engine = SentimentEngine()

    # ------------------------------------------------------------------ #
    # Data Fetching (unchanged interface, same resilience)               #
    # ------------------------------------------------------------------ #

    async def fetch_market_data(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 250,
    ) -> pd.DataFrame:
        """Fetch market bars with caching, retries, and extended lookback."""
        cache_key = f"{symbol}_{timeframe}"
        now = time.time()

        if symbol == "SPY" and cache_key in self._cache:
            entry = self._cache[cache_key]
            if now - entry["timestamp"] < self._CACHE_TTL:
                return entry["data"]

        df = await self._fetch_with_backoff(symbol, timeframe, limit)

        if not df.empty and symbol == "SPY":
            self._cache[cache_key] = {"timestamp": now, "data": df}

        return df

    async def _fetch_with_backoff(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        bars = []
        for attempt in range(2):
            try:
                days_to_lookback = (limit // 390) + 3
                start_date = (
                    datetime.now(timezone.utc) - timedelta(days=days_to_lookback)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")

                bars = await self.client.get_bars(
                    symbol, timeframe=timeframe, limit=limit, start=start_date
                )
                if len(bars) >= 20:
                    break
                if attempt == 0:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"Alpaca fetch failed for {symbol} (Attempt {attempt}): {e}")

        df = None
        if bars and len(bars) >= 5:
            df = pd.DataFrame(bars)
            mapping = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        else:
            try:
                import yfinance as yf

                yf_logger = logging.getLogger("yfinance")
                old_yf_level = yf_logger.level
                yf_logger.setLevel(logging.CRITICAL)
                interval = "1d" if timeframe == "1D" else "15m" if timeframe == "15Min" else "1m"
                df = yf.download(symbol, period="7d", interval=interval, progress=False)
                if getattr(df, "empty", True) and timeframe == "1Min":
                    df = yf.download(symbol, period="7d", interval="15m", progress=False)
            except Exception as e:
                logger.debug(f"yfinance fallback failed for {symbol}: {e}")
                return pd.DataFrame()
            finally:
                try:
                    if "yf_logger" in locals() and "old_yf_level" in locals():
                        yf_logger.setLevel(old_yf_level)
                except Exception:
                    pass

        if df is None or (hasattr(df, "empty") and df.empty):
            return pd.DataFrame()

        return self._normalize_columns(df, limit)

    def _normalize_columns(self, df: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Standardize column names across different data providers."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        rename_map = {"adj close": "close", "unadjusted close": "close"}
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = df["close"] if "close" in df.columns else 0.0

            # Type check workaround for pandas multi-column assignment
            col_data = df[col]
            if isinstance(col_data, pd.DataFrame):
                df[col] = col_data.iloc[:, 0]

        return df.loc[:, ~df.columns.duplicated()].tail(limit)

    # ------------------------------------------------------------------ #
    # Superhuman Signal Generation                                        #
    # ------------------------------------------------------------------ #

    def generate_signal(self, data: pd.DataFrame, sentiment_score: float = 0.0) -> float:
        """
        Generate multi-factor alpha signal with Superhuman intelligence.

        Factor stack:
        1. Adaptive Kalman trend score (regime-adaptive noise filter)
        2. Entropy-filtered momentum (suppresses noise via Shannon entropy)
        3. Hurst-gated direction gate (momentum vs. mean-reversion)
        4. VWAP deviation signal (smart money flow detection)
        5. Hawkes intensity adjustment (vol clustering risk penalty)
        """
        if data.empty or "close" not in data.columns:
            return 0.0

        prices = data["close"].astype(float).to_numpy().flatten()

        # Adapt Kalman filter to current volatility environment
        if len(prices) > 5:
            rolling_vol = float(np.std(np.diff(prices[-20:]) / (prices[-20:-1] + 1e-9)))
            self.kf.adapt_to_volatility(rolling_vol)

        denoised_prices = self.kf.batch_filter(prices)
        if len(denoised_prices) < 5:
            return 0.0

        pct_changes = pd.Series(denoised_prices).pct_change().dropna()
        if pct_changes.empty:
            return 0.0

        # --- Factor 1: Trend Signal (Kalman-smoothed) ---
        trend = (
            float(denoised_prices[-1] / denoised_prices[-10] - 1)
            if len(denoised_prices) >= 10
            else 0.0
        )
        trend_score = float(np.tanh(trend * 20))

        # --- Factor 2: Entropy-Filtered Momentum ---
        momentum = float(pct_changes.tail(5).mean())
        volatility = (
            float(pct_changes.tail(20).std())
            if len(pct_changes) >= 20
            else float(pct_changes.std())
        )
        entropy_filter = self._compute_entropy_filter(pct_changes.to_numpy())
        momentum_score = float(np.tanh(momentum * 10)) * entropy_filter

        # --- Factor 3: Hurst Exponent Gate ---
        price_series = pd.Series(prices)
        hurst = compute_hurst_exponent(price_series)
        hurst_signal = self._hurst_gate(hurst, trend_score, momentum_score)

        # --- Factor 4: VWAP Deviation Signal ---
        vwap_signal = self._compute_vwap_signal(data, prices[-1] if len(prices) > 0 else 0.0)

        # --- Factor 5: Hawkes Intensity Adjustment ---
        pct_arr = pct_changes.to_numpy()
        extreme_mask = np.abs(pct_arr) > pct_changes.std()
        extreme_idx = np.where(extreme_mask)[0].astype(float)
        intensity = self.hawkes.calculate_intensity(extreme_idx)
        hawkes_adj = 1.0 / (1.0 + intensity)

        # --- Volatility Penalty ---
        vol_penalty = 1.0 / (1.0 + volatility * 8.0)

        # --- Factor 6: News Sentiment ---
        # Sentiment contributes depending on config
        sentiment_adj = 0.0
        if Config.SENTIMENT_ENABLED:
            sentiment_adj = sentiment_score * Config.SENTIMENT_WEIGHT

        # --- IC-Optimal Factor Blending ---
        # Weights: Hurst-gated direction (0.35) + entropy-mom (0.30) + trend (0.20) + VWAP (0.15)
        # We scale base signal slightly to make room for sentiment
        base_signal = (
            (0.35 * hurst_signal + 0.30 * momentum_score + 0.20 * trend_score + 0.15 * vwap_signal)
            * vol_penalty
            * hawkes_adj
        )

        signal = base_signal * (1.0 - Config.SENTIMENT_WEIGHT) + sentiment_adj

        return float(np.clip(signal, -1.0, 1.0))

    def _compute_entropy_filter(self, returns: np.ndarray[Any, Any]) -> float:
        """
        Shannon entropy filter for momentum signals.

        High entropy (random-looking returns) → filter = 0.3 (suppress signal)
        Low entropy (directional returns) → filter = 1.0 (pass signal)
        """
        if len(returns) < 5:
            return 0.6
        try:
            hist, _ = np.histogram(returns, bins=min(10, len(returns) // 2), density=True)
            hist = hist[hist > 0]
            if len(hist) == 0:
                return 0.6
            entropy = float(-np.sum(hist * np.log(hist + 1e-9)))
            # Low entropy (< 1.5) = directional; High entropy (> 3.5) = noisy
            clip_val = float(np.clip((entropy - 1.5) / 2.0, 0.0, 0.70))
            filter_val = 1.0 - clip_val
            return float(filter_val)
        except Exception:
            return 0.6

    def _hurst_gate(self, hurst: float, trend_score: float, momentum_score: float) -> float:
        """
        Hurst-gated directional signal.

        H > 0.60 → Persistent trend → use momentum direction, amplified
        H < 0.40 → Mean-reverting → flip signal direction
        H ≈ 0.50 → Random walk → suppress signal (no statistical edge)
        """
        if hurst > 0.60:
            # Persistent: trend and momentum agree → amplify
            amplifier = min(1.5, 1.0 + (hurst - 0.60) * 3.0)
            return float(np.tanh((trend_score * 0.55 + momentum_score * 0.45) * amplifier))
        elif hurst < 0.40:
            # Mean-reverting: opposite of momentum
            reverter = min(1.3, 1.0 + (0.40 - hurst) * 2.5)
            return float(np.tanh(-momentum_score * reverter))
        else:
            # Random walk zone: very weak signal
            random_walk_suppressor = 1.0 - abs(hurst - 0.50) * 4.0
            return float(np.tanh((trend_score + momentum_score) * 0.5 * random_walk_suppressor))

    def _compute_vwap_signal(self, data: pd.DataFrame, current_price: float) -> float:
        """
        VWAP deviation signal for institutional flow detection.

        Price above VWAP with rising volume → institutional accumulation (bullish)
        Price below VWAP with rising volume → institutional distribution (bearish)
        """
        if "volume" not in data.columns or "close" not in data.columns:
            return 0.0
        if len(data) < 5:
            return 0.0

        try:
            close = data["close"].astype(float)
            volume = data["volume"].astype(float)

            typical_price = close
            if "high" in data.columns and "low" in data.columns:
                typical_price = (
                    data["high"].astype(float) + data["low"].astype(float) + close
                ) / 3.0

            vwap_num = (typical_price * volume).cumsum()
            vwap_den = volume.cumsum().replace(0, np.nan)
            vwap = (vwap_num / vwap_den).dropna()

            if vwap.empty:
                return 0.0

            vwap_val = float(vwap.iloc[-1])
            deviation = (current_price - vwap_val) / max(vwap_val, 1e-6)

            # Volume trend
            vol_avg = float(volume.tail(20).mean())
            vol_recent = float(volume.tail(5).mean())
            vol_ratio = vol_recent / max(vol_avg, 1.0)

            # Amplify when volume confirms the deviation
            vol_confirmation = min(1.5, vol_ratio)
            signal = float(np.tanh(deviation * 15.0 * vol_confirmation))
            return signal
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # Monte Carlo (unchanged interface, improved sampling)               #
    # ------------------------------------------------------------------ #

    def monte_carlo_simulation(
        self,
        prices: np.ndarray[Any, Any],
        num_paths: int = 500,
        horizon: int = 20,
    ) -> float:
        """
        Antithetic variate Monte Carlo for variance reduction.
        Approximately doubles statistical efficiency vs. naive sampling.
        """
        prices = prices.flatten()
        if len(prices) < 5:
            return 0.5

        returns = np.diff(np.log(prices)).astype(float)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns)) if np.std(returns) > 1e-9 else 0.01
        last_price = float(prices[-1])
        success_count = 0
        half_paths = num_paths // 2

        rng = np.random.default_rng()
        for _ in range(half_paths):
            # Generate path + antithetic path
            sampled = rng.normal(mu, sigma, horizon)
            antithetic = (2.0 * mu) - sampled  # antithetic variates

            for path in [sampled, antithetic]:
                final_price = last_price * np.exp(float(np.sum(path)))
                if final_price > last_price:
                    success_count += 1

        return float(success_count / num_paths)

    # ------------------------------------------------------------------ #
    # Batch Signal Generation                                             #
    # ------------------------------------------------------------------ #

    async def get_batch_signals(
        self, symbols: List[str], timeframe: str = "15Min"
    ) -> Dict[str, float]:
        """
        Generate Superhuman alpha signals for all symbols.
        Blends: Kalman+Entropy+Hurst signal (0.65) + Monte Carlo (0.35)
        Now uses multi-timeframe fetching (1Min, 15Min, 1D) for robust signal.
        """
        signals: Dict[str, float] = {}
        semaphore = asyncio.Semaphore(2)

        async def symbol_signal(symbol: str) -> Tuple[str, float]:
            async with semaphore:
                # Fetch multi-timeframe data
                data_15m = await self.fetch_market_data(symbol, timeframe="15Min")
                data_1m = await self.fetch_market_data(symbol, timeframe="1Min")
                data_1d = await self.fetch_market_data(symbol, timeframe="1D", limit=100)

                sentiment = await self.sentiment_engine.get_sentiment(symbol)

                alpha_15m = self.generate_signal(data_15m, sentiment) if not data_15m.empty else 0.0
                alpha_1m = self.generate_signal(data_1m, sentiment) if not data_1m.empty else 0.0
                alpha_1d = self.generate_signal(data_1d, sentiment) if not data_1d.empty else 0.0

                # Multi-timeframe blending
                alpha = (
                    alpha_1m * Config.SIGNAL_1MIN_WEIGHT
                    + alpha_15m * Config.SIGNAL_15MIN_WEIGHT
                    + alpha_1d * Config.SIGNAL_1D_WEIGHT
                )

                if not data_15m.empty:
                    mc_prob = self.monte_carlo_simulation(
                        data_15m["close"].astype(float).to_numpy()
                    )
                    # IC-optimal blending: signal carries more weight than MC
                    alpha = alpha * 0.65 + (mc_prob - 0.5) * 0.70
                return symbol, float(np.clip(alpha, -1.0, 1.0))

        results = await asyncio.gather(*[symbol_signal(s) for s in symbols], return_exceptions=True)
        for r in results:
            if isinstance(r, tuple):
                signals[r[0]] = r[1]
        return signals

    async def close(self) -> None:
        if hasattr(self.client, "close"):
            try:
                await self.client.close()
            except Exception as exc:
                logger.warning(f"Failed to close AlphaEngine client: {exc}")
