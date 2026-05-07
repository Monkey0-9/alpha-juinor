import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

    async def fetch_market_data(self, symbol: str, timeframe: str = "1Min", limit: int = 120) -> pd.DataFrame:
        """Fetch market bars from Alpaca directly, with yfinance fallback."""
        bars = []
        try:
            if timeframe == "1D":
                start_date = (datetime.utcnow() - timedelta(days=limit)).strftime("%Y-%m-%d")
                bars = await self.client.get_bars(symbol, timeframe=timeframe, limit=limit, start=start_date)
            else:
                bars = await self.client.get_bars(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            logger.debug(f"Alpaca data fetch failed for {symbol} ({timeframe}): {e}")

        if bars:
            df = pd.DataFrame(bars)
            if "close" not in df.columns and "c" in df.columns:
                df["close"] = df["c"]
            return df
            
        # Fallback to yfinance if Alpaca fails (e.g., invalid keys or rate limits)
        logger.debug(f"Falling back to yfinance for {symbol} data...")
        try:
            import yfinance as yf
            interval = "1d" if timeframe == "1D" else "15m" if timeframe == "15Min" else "1m"
            period = f"{limit}d" if timeframe == "1D" else "5d"
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = df.columns.str.lower()
                if "close" not in df.columns and "adj close" in df.columns:
                    df["close"] = df["adj close"]
                return df.tail(limit)
        except Exception as e:
            logger.warning(f"yfinance fallback failed for {symbol}: {e}")

        return pd.DataFrame()
    def generate_signal(self, data: pd.DataFrame) -> float:
        """Generate alpha signal from market data."""
        if data.empty or "close" not in data.columns:
            return 0.0

        prices = data["close"].astype(float).to_numpy()
        denoised_prices = self.kf.batch_filter(prices)

        returns = pd.Series(denoised_prices).pct_change().dropna()
        burst_threshold = returns.std() * 1.5
        mask = abs(returns) > burst_threshold
        burst_events = np.array(returns[mask].index, dtype=float)

        intensity = 0.1
        if len(burst_events) > 0:
            intensity = self.hawkes.calculate_intensity(burst_events)

        recent_trend = 0.0
        if len(denoised_prices) >= 5:
            recent_trend = (denoised_prices[-1] / denoised_prices[-5]) - 1

        vol_scaler = 1.0 / (1.0 + intensity)
        signal = np.tanh(recent_trend * 40) * vol_scaler
        return float(signal)

    async def get_batch_signals(
        self,
        symbols: List[str],
        timeframe: str = "15Min"
    ) -> Dict[str, float]:
        """Generate signals for a batch of symbols."""
        signals: Dict[str, float] = {}

        async def symbol_signal(symbol: str) -> tuple:
            data = await self.fetch_market_data(
                symbol,
                timeframe=timeframe,
                limit=120
            )
            return symbol, self.generate_signal(data)

        tasks = [symbol_signal(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(symbols, results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"Signal generation failed for {symbol}: {result}"
                )
                signals[symbol] = 0.0
            else:
                signals[symbol] = result[1]

        return signals
