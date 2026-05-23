import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Tuple
from nexus.math.models import KalmanFilter
from nexus.math.indicators import HawkesProcess
from nexus.execution.alpaca import get_client
from nexus.utils.config import Config

logger = logging.getLogger(__name__)


class AlphaEngine:
    """Alpha Generation Engine with robust data resilience and caching."""

    _cache: Dict[str, Dict[str, Any]] = {}
    _CACHE_TTL = 300  # 5 minutes for benchmark data

    def __init__(self, backend_url: str = Config.BACKEND_URL):
        self.kf = KalmanFilter()
        self.hawkes = HawkesProcess()
        self.backend_url = backend_url
        self.client = get_client()

    async def fetch_market_data(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 250  # Increased default for regime detection
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
        # Institutional Resilience: Try Alpaca with extended lookback if needed
        for attempt in range(2):
            try:
                # If we need many bars and it's 1Min, we need to specify a start date
                # to get data even if market was closed recently.
                days_to_lookback = (limit // 390) + 3 # approx 390 mins in a trading day
                start_date = (datetime.now(timezone.utc) - timedelta(days=days_to_lookback)).strftime("%Y-%m-%dT%H:%M:%SZ")
                
                bars = await self.client.get_bars(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                    start=start_date
                )
                if len(bars) >= 20: # Enough for regime detection
                    break
                if attempt == 0:
                    await asyncio.sleep(1) # Transient wait
            except Exception as e:
                logger.debug(f"Alpaca fetch failed for {symbol} (Attempt {attempt}): {e}")

        df = None
        if bars and len(bars) >= 5:
            df = pd.DataFrame(bars)
            mapping = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        else:
            # yfinance Fallback with robust period
            try:
                import yfinance as yf
                yf_logger = logging.getLogger("yfinance")
                old_yf_level = yf_logger.level
                yf_logger.setLevel(logging.CRITICAL)
                interval = "1d" if timeframe == "1D" else "15m" if timeframe == "15Min" else "1m"
                # Always ask for at least 7 days to ensure we get 20+ bars
                df = yf.download(symbol, period="7d", interval=interval, progress=False)
                if df.empty and timeframe == "1Min":
                    # Try 15m if 1m is unavailable (e.g. too old)
                    df = yf.download(symbol, period="7d", interval="15m", progress=False)
            except Exception as e:
                logger.debug(f"yfinance fallback failed for {symbol}: {e}")
                return pd.DataFrame()
            finally:
                try:
                    yf_logger.setLevel(old_yf_level)
                except Exception:
                    pass

        if df is None or df.empty:
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
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        return df.loc[:, ~df.columns.duplicated()].tail(limit)

    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate alpha signal from market data."""
        if data.empty or "close" not in data.columns:
            return 0.0

        prices = data["close"].astype(float).to_numpy().flatten()
        denoised_prices = self.kf.batch_filter(prices)

        if len(denoised_prices) < 5:
            return 0.0

        pct_changes = pd.Series(denoised_prices).pct_change().dropna()
        if pct_changes.empty:
            return 0.0

        momentum = float(pct_changes.tail(5).mean())
        volatility = float(pct_changes.tail(20).std()) if len(pct_changes) >= 20 else float(pct_changes.std())
        trend = float(denoised_prices[-1] / denoised_prices[-10] - 1) if len(denoised_prices) >= 10 else momentum

        intensity = self.hawkes.calculate_intensity(
            np.array(pct_changes[abs(pct_changes) > pct_changes.std() * 1.5].index, dtype=float)
        )

        trend_score = np.tanh(trend * 20)
        momentum_score = np.tanh(momentum * 10)
        volatility_penalty = 1.0 / (1.0 + volatility * 8.0)
        hawkes_adjustment = 1.0 / (1.0 + intensity)

        signal = (0.45 * trend_score + 0.35 * momentum_score) * volatility_penalty * hawkes_adjustment
        return float(np.clip(signal, -1.0, 1.0))

    def monte_carlo_simulation(self, prices: np.ndarray[Any, Any], num_paths: int = 200, horizon: int = 20) -> float:
        prices = prices.flatten()
        if len(prices) < 5:
            return 0.5
        returns = np.diff(np.log(prices))
        empirical = returns.astype(float)
        success_count = 0
        last_price = float(prices[-1])

        for _ in range(num_paths):
            sampled = np.random.choice(empirical, size=horizon, replace=True)
            final_price = last_price * np.exp(np.sum(sampled))
            if final_price > last_price:
                success_count += 1

        return float(success_count / num_paths)

    async def get_batch_signals(self, symbols: List[str], timeframe: str = "15Min") -> Dict[str, float]:
        signals: Dict[str, float] = {}
        semaphore = asyncio.Semaphore(2)  # lower concurrency to avoid Alpaca data throttling

        async def symbol_signal(symbol: str) -> Tuple[str, float]:
            async with semaphore:
                data = await self.fetch_market_data(symbol, timeframe=timeframe)
                alpha = self.generate_signal(data)
                if not data.empty:
                    alpha = alpha * 0.7 + (self.monte_carlo_simulation(data["close"].astype(float).to_numpy()) - 0.5) * 0.6
                return symbol, alpha

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
