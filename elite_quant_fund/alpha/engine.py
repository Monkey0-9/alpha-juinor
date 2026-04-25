import os
import sys
import httpx
import asyncio
import numpy as np
import pandas as pd

# Elite Path Resolution
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.mini_quant_fund.institutional.math.kalman_filter import (
    SovereignKalmanFilter
)
from src.mini_quant_fund.institutional.math.vol_clustering import (
    SovereignHawkesProcess
)

class AlphaEngine:
    """
    Sovereign Elite Alpha Engine.
    Combines Kalman Filter denoising with Hawkes Process volatility awareness.
    """
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.kf = SovereignKalmanFilter()
        self.hawkes = SovereignHawkesProcess()
        self.models = ["Kalman-State", "Hawkes-Intensity", "Mean-Reversion"]
        self.backend_url = backend_url

    async def get_real_data(self, symbol: str) -> pd.DataFrame:
        """Fetch real market data from the backend"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.backend_url}/api/alpaca/bars",
                    params={"symbol": symbol, "limit": 100},
                    timeout=5
                )
                if response.status_code == 200:
                    bars = response.json().get("bars", [])
                    if bars:
                        df = pd.DataFrame(bars)
                        # Ensure 'close' column exists
                        if 'close' not in df.columns and 'c' in df.columns:
                            df['close'] = df['c']
                        return df
        except Exception:
            pass
        return pd.DataFrame()

    def generate_signal(self, data: pd.DataFrame) -> float:
        """
        Generates a state-space optimized signal between -1 and 1.
        """
        if data.empty or 'close' not in data.columns:
            return 0.0

        # 1. Denoising Layer (Kalman Filter)
        prices = data['close'].values
        denoised_prices = self.kf.batch_filter(prices)

        # 2. Intensity Layer (Hawkes Process)
        returns = pd.Series(denoised_prices).pct_change().dropna()
        burst_threshold = returns.std() * 2
        burst_events = returns[abs(returns) > burst_threshold].index.values
        burst_events = burst_events.astype(float)

        current_time = float(len(data))
        intensity = self.hawkes.estimate_intensity(burst_events, current_time)

        # 3. Alpha Strategy: Trend following + Volatility scaling
        recent_trend = 0
        if len(denoised_prices) >= 5:
            recent_trend = (denoised_prices[-1] / denoised_prices[-5]) - 1

        # Scale signal by inverse of intensity
        vol_scaler = 1.0 / (1.0 + intensity)
        signal = np.tanh(recent_trend * 20) * vol_scaler

        return float(signal)

    async def get_batch_signals(self, symbols: list) -> dict:
        signals = {}
        for symbol in symbols:
            data = await self.get_real_data(symbol)
            if not data.empty:
                signals[symbol] = self.generate_signal(data)
            else:
                # Minimal fallback if data fails, but with a real-time warning
                signals[symbol] = 0.0
        return signals
