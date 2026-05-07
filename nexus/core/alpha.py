import asyncio
import logging
import httpx
import numpy as np
import pandas as pd
from typing import Dict, List
from nexus.math.models import KalmanFilter
from nexus.math.indicators import HawkesProcess
from nexus.utils.config import Config

logger = logging.getLogger(__name__)


class AlphaEngine:
    """Alpha Generation Engine using multi-timeframe signals."""

    def __init__(self, backend_url: str = Config.BACKEND_URL):
        self.kf = KalmanFilter()
        self.hawkes = HawkesProcess()
        self.backend_url = backend_url

    async def fetch_market_data(self, symbol: str, timeframe: str = "1Min", limit: int = 120) -> pd.DataFrame:
        """Fetch market bars from the execution backend."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.backend_url}/api/alpaca/bars",
                    params={"symbol": symbol, "timeframe": timeframe, "limit": limit},
                    timeout=10
                )
                if response.status_code == 200:
                    bars = response.json().get("bars", [])
                    if bars:
                        df = pd.DataFrame(bars)
                        if "close" not in df.columns and "c" in df.columns:
                            df["close"] = df["c"]
                        return df
        except Exception as e:
            logger.debug(f"Data fetch failed for {symbol} ({timeframe}): {e}")
        return pd.DataFrame()

    def generate_signal(self, data: pd.DataFrame) -> float:
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

    async def get_batch_signals(self, symbols: List[str], timeframe: str = "15Min") -> Dict[str, float]:
        signals: Dict[str, float] = {}

        async def symbol_signal(symbol: str) -> tuple[str, float]:
            data = await self.fetch_market_data(symbol, timeframe=timeframe, limit=120)
            return symbol, self.generate_signal(data)

        tasks = [symbol_signal(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(symbols, results):
            if isinstance(result, BaseException):
                logger.warning(f"Signal generation failed for {symbol}: {result}")
                signals[symbol] = 0.0
            else:
                signals[symbol] = result[1]

        return signals
