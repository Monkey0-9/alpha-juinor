import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from nexus.math.models import KalmanFilter
from nexus.math.indicators import HawkesProcess
from nexus.execution.alpaca import get_client
from nexus.utils.config import Config

logger = logging.getLogger(__name__)


class AlphaEngine:
    """Alpha Generation Engine using multi-timeframe signals."""

    def __init__(self, backend_url: str = Config.BACKEND_URL):
        self.kf = KalmanFilter()
        self.hawkes = HawkesProcess()
        self.backend_url = backend_url
        self.client = get_client()

    async def fetch_market_data(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 120
    ) -> pd.DataFrame:
        """Fetch market bars from Alpaca directly, with yfinance fallback."""
        bars = []
        try:
            if timeframe == "1D":
                start_date = (
                    datetime.now(timezone.utc) - timedelta(days=limit)
                ).strftime("%Y-%m-%d")
                bars = await self.client.get_bars(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                    start=start_date
                )
            else:
                bars = await self.client.get_bars(
                    symbol,
                    timeframe=timeframe,
                    limit=limit
                )
        except Exception as e:
            logger.debug(f"Alpaca data fetch failed for {symbol}: {e}")

        df = None
        if bars:
            df = pd.DataFrame(bars)
            mapping = {
                "o": "open", "h": "high", "l": "low", "c": "close",
                "v": "volume", "vw": "vwap", "n": "trade_count"
            }
            df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
        else:
            logger.debug(f"Falling back to yfinance for {symbol} data...")
            try:
                import yfinance as yf
                interval = "1d" if timeframe == "1D" else "15m" if timeframe == "15Min" else "1m"
                period = f"{limit}d" if timeframe == "1D" else "5d"
                df = yf.download(symbol, period=period, interval=interval, progress=False)
            except Exception as e:
                logger.warning(f"yfinance fallback failed for {symbol}: {e}")
                return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # ROBUST COLUMN NORMALIZATION
        # 1. Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]).lower() for c in df.columns]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # 2. Standardize common names
        rename_map = {
            "adj close": "close",
            "unadjusted close": "close",
            "volume": "volume",
        }
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]

        # 3. Ensure mandatory columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                if "close" in df.columns:
                    df[col] = df["close"]
                elif "c" in df.columns:
                    df[col] = df["c"]
                else:
                    df[col] = 0.0

        # 4. Final Cleanup: Drop duplicates and squeeze to ensure single columns
        df = df.loc[:, ~df.columns.duplicated()]
        # Ensure all core columns are Series, not DataFrames
        for col in ["open", "high", "low", "close", "volume"]:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

        return df.tail(limit)

    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate alpha signal from market data."""
        if data.empty or "close" not in data.columns:
            return 0.0

        # Squeeze to ensure 1D if it was somehow 2D
        prices = data["close"].astype(float).to_numpy()
        if prices.ndim > 1:
            prices = prices.flatten()
        denoised_prices = self.kf.batch_filter(prices)

        returns = pd.Series(denoised_prices).pct_change().dropna()
        if returns.empty:
            return 0.0
            
        burst_threshold = returns.std() * 1.5
        mask = abs(returns) > burst_threshold
        burst_events = np.array(returns[mask].index, dtype=float)

        intensity = 0.1
        if len(burst_events) > 0:
            intensity = self.hawkes.calculate_intensity(burst_events)

        recent_trend = 0.0
        if len(denoised_prices) >= 5:
            # Ensure scalars
            p_last = float(denoised_prices[-1])
            p_prev = float(denoised_prices[-5])
            recent_trend = (p_last / p_prev) - 1

        vol_scaler = 1.0 / (1.0 + intensity)
        signal = np.tanh(recent_trend * 40) * vol_scaler
        return float(signal)

    def monte_carlo_simulation(
        self,
        prices: np.ndarray,
        num_paths: int = 100,
        horizon: int = 20
    ) -> float:
        """Perform Monte Carlo simulation for price path forecasting."""
        # Ensure 1D
        if prices.ndim > 1:
            prices = prices.flatten()
        if len(prices) < 2:
            return 0.5
        returns = np.diff(np.log(prices))
        mu = np.mean(returns)
        sigma = np.std(returns)

        last_price = float(prices[-1])
        sim_results = []

        for _ in range(num_paths):
            path_last = last_price
            for _ in range(horizon):
                path_last = path_last * np.exp(mu + sigma * np.random.normal())
            sim_results.append(path_last)

        prob_up = sum(1 for p in sim_results if p > last_price) / num_paths
        return float(prob_up)

    async def get_batch_signals(
        self,
        symbols: List[str],
        timeframe: str = "15Min"
    ) -> Dict[str, float]:
        """Generate signals for a batch of symbols with concurrency control."""
        signals: Dict[str, float] = {}
        # Institutional Concurrency Control
        semaphore = asyncio.Semaphore(50)

        async def symbol_signal(symbol: str) -> tuple:
            async with semaphore:
                await asyncio.sleep(0.1) # Rate limit protection
                data = await self.fetch_market_data(
                    symbol,
                    timeframe=timeframe,
                    limit=120
                )
                alpha = self.generate_signal(data)

                # Quant Computer Layer: Path Simulation
                if not data.empty:
                    prices = data["close"].astype(float).to_numpy()
                    mc_prob = self.monte_carlo_simulation(prices)
                    alpha = alpha * 0.7 + (mc_prob - 0.5) * 0.6

                return symbol, alpha

        tasks = [symbol_signal(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple):
                symbol, alpha = result
                signals[symbol] = alpha
            elif isinstance(result, BaseException):
                logger.debug(f"Batch signal exception: {result}")

        return signals
