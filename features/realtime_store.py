"""
Real-Time Feature Store - Streaming Feature Calculation.

Features:
- Sub-second feature updates
- Streaming price ingestion
- Feature caching layer
- Event-driven updates
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """Single market tick."""
    symbol: str
    timestamp: float
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class StreamingFeatures:
    """Real-time calculated features."""
    symbol: str
    timestamp: float

    # Price features
    last_price: float = 0.0
    vwap: float = 0.0

    # Return features
    return_1m: float = 0.0
    return_5m: float = 0.0
    return_15m: float = 0.0

    # Momentum features
    momentum_score: float = 0.0
    trend_strength: float = 0.0

    # Volatility features
    realized_vol_1h: float = 0.0
    volatility_ratio: float = 0.0

    # Volume features
    volume_ratio: float = 1.0
    cumulative_volume: int = 0

    # Technical features
    rsi_14: float = 50.0
    macd_signal: float = 0.0
    bollinger_position: float = 0.0

    # Microstructure
    spread_bps: float = 0.0
    trade_imbalance: float = 0.0


class SymbolBuffer:
    """Circular buffer for streaming calculations."""

    def __init__(self, max_ticks: int = 10000):
        self.max_ticks = max_ticks
        self.ticks = deque(maxlen=max_ticks)
        self.prices = deque(maxlen=max_ticks)
        self.volumes = deque(maxlen=max_ticks)
        self.timestamps = deque(maxlen=max_ticks)

        # Aggregated data
        self.minute_bars: Dict[int, Dict] = {}
        self.cumulative_volume = 0
        self.cumulative_value = 0.0

    def add_tick(self, tick: TickData):
        """Add tick to buffer."""
        self.ticks.append(tick)
        self.prices.append(tick.price)
        self.volumes.append(tick.volume)
        self.timestamps.append(tick.timestamp)

        self.cumulative_volume += tick.volume
        self.cumulative_value += tick.price * tick.volume

        # Update minute bar
        minute = int(tick.timestamp // 60)
        if minute not in self.minute_bars:
            self.minute_bars[minute] = {
                "open": tick.price,
                "high": tick.price,
                "low": tick.price,
                "close": tick.price,
                "volume": 0
            }

        bar = self.minute_bars[minute]
        bar["high"] = max(bar["high"], tick.price)
        bar["low"] = min(bar["low"], tick.price)
        bar["close"] = tick.price
        bar["volume"] += tick.volume

    def get_vwap(self) -> float:
        """Calculate VWAP."""
        if self.cumulative_volume == 0:
            return 0.0
        return self.cumulative_value / self.cumulative_volume

    def get_returns(self, lookback_seconds: int) -> List[float]:
        """Get returns for lookback period."""
        if len(self.prices) < 2:
            return []

        cutoff = time.time() - lookback_seconds
        prices = []

        for i, ts in enumerate(self.timestamps):
            if ts >= cutoff:
                prices.append(self.prices[i])

        if len(prices) < 2:
            return []

        return [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]

    def get_prices_array(self, n: int = 100) -> np.ndarray:
        """Get last n prices as array."""
        prices = list(self.prices)
        return np.array(prices[-n:])


class RealTimeFeatureStore:
    """
    Real-time streaming feature store.

    Provides sub-second feature updates for trading decisions.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        feature_update_interval_ms: int = 100
    ):
        self.buffer_size = buffer_size
        self.update_interval = feature_update_interval_ms / 1000

        # Symbol buffers
        self.buffers: Dict[str, SymbolBuffer] = {}

        # Cached features
        self.features: Dict[str, StreamingFeatures] = {}

        # Callbacks
        self.callbacks: List[Callable] = []

        # Background thread
        self._running = False
        self._thread = None

        # Baseline volatility (for comparison)
        self.baseline_vol: Dict[str, float] = {}

    def start(self):
        """Start background feature calculation."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._feature_loop, daemon=True)
        self._thread.start()
        logger.info("Real-time feature store started")

    def stop(self):
        """Stop background processing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _feature_loop(self):
        """Background feature calculation loop."""
        while self._running:
            try:
                for symbol in list(self.buffers.keys()):
                    self._update_features(symbol)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Feature loop error: {e}")

    def ingest_tick(self, tick: TickData):
        """Ingest a new market tick."""
        symbol = tick.symbol

        if symbol not in self.buffers:
            self.buffers[symbol] = SymbolBuffer(self.buffer_size)

        self.buffers[symbol].add_tick(tick)

    def ingest_batch(self, ticks: List[TickData]):
        """Ingest batch of ticks."""
        for tick in ticks:
            self.ingest_tick(tick)

    def _update_features(self, symbol: str):
        """Update features for a symbol."""
        if symbol not in self.buffers:
            return

        buffer = self.buffers[symbol]

        if len(buffer.prices) < 10:
            return

        prices = buffer.get_prices_array()
        current_price = prices[-1]

        # Calculate returns
        returns_1m = buffer.get_returns(60)
        returns_5m = buffer.get_returns(300)
        returns_15m = buffer.get_returns(900)

        return_1m = sum(returns_1m) if returns_1m else 0.0
        return_5m = sum(returns_5m) if returns_5m else 0.0
        return_15m = sum(returns_15m) if returns_15m else 0.0

        # Momentum score
        momentum = (return_1m * 0.5 + return_5m * 0.3 + return_15m * 0.2) * 100
        momentum = float(np.clip(momentum, -1, 1))

        # Volatility
        returns_1h = buffer.get_returns(3600)
        vol_1h = np.std(returns_1h) * np.sqrt(252 * 6.5) if len(returns_1h) > 10 else 0.02

        baseline = self.baseline_vol.get(symbol, vol_1h)
        vol_ratio = vol_1h / (baseline + 0.001)

        # Volume ratio
        recent_vol = sum(list(buffer.volumes)[-100:]) if len(buffer.volumes) >= 100 else 0
        avg_vol = buffer.cumulative_volume / max(len(buffer.volumes), 1) * 100
        volume_ratio = recent_vol / (avg_vol + 1) if avg_vol > 0 else 1.0

        # RSI (simplified streaming version)
        if len(prices) >= 15:
            gains = []
            losses = []
            for i in range(-14, 0):
                delta = prices[i] - prices[i-1]
                if delta > 0:
                    gains.append(delta)
                else:
                    losses.append(abs(delta))

            avg_gain = np.mean(gains) if gains else 0.001
            avg_loss = np.mean(losses) if losses else 0.001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50.0

        # Bollinger position
        if len(prices) >= 20:
            ma = np.mean(prices[-20:])
            std = np.std(prices[-20:])
            upper = ma + 2 * std
            lower = ma - 2 * std
            bb_pos = (current_price - lower) / (upper - lower + 0.001)
        else:
            bb_pos = 0.5

        # Spread
        last_tick = buffer.ticks[-1] if buffer.ticks else None
        if last_tick and last_tick.bid and last_tick.ask:
            spread_bps = (last_tick.ask - last_tick.bid) / current_price * 10000
        else:
            spread_bps = 5.0

        # Update features
        self.features[symbol] = StreamingFeatures(
            symbol=symbol,
            timestamp=time.time(),
            last_price=current_price,
            vwap=buffer.get_vwap(),
            return_1m=return_1m,
            return_5m=return_5m,
            return_15m=return_15m,
            momentum_score=momentum,
            trend_strength=abs(momentum),
            realized_vol_1h=vol_1h,
            volatility_ratio=float(vol_ratio),
            volume_ratio=float(volume_ratio),
            cumulative_volume=buffer.cumulative_volume,
            rsi_14=float(rsi),
            bollinger_position=float(bb_pos),
            spread_bps=float(spread_bps)
        )

        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(symbol, self.features[symbol])
            except Exception as e:
                logger.debug(f"Callback error: {e}")

    def get_features(self, symbol: str) -> Optional[StreamingFeatures]:
        """Get current features for symbol."""
        return self.features.get(symbol)

    def get_all_features(self) -> Dict[str, StreamingFeatures]:
        """Get all current features."""
        return self.features.copy()

    def register_callback(self, callback: Callable):
        """Register callback for feature updates."""
        self.callbacks.append(callback)

    def set_baseline_volatility(self, symbol: str, vol: float):
        """Set baseline volatility for comparison."""
        self.baseline_vol[symbol] = vol

    def get_feature_vector(self, symbol: str) -> Optional[np.ndarray]:
        """Get feature vector for ML models."""
        features = self.get_features(symbol)
        if not features:
            return None

        return np.array([
            features.return_1m,
            features.return_5m,
            features.return_15m,
            features.momentum_score,
            features.realized_vol_1h,
            features.volatility_ratio,
            features.volume_ratio,
            features.rsi_14 / 100,
            features.bollinger_position,
            features.spread_bps / 100
        ])


# Global singleton
_store: Optional[RealTimeFeatureStore] = None


def get_realtime_feature_store() -> RealTimeFeatureStore:
    """Get or create global real-time feature store."""
    global _store
    if _store is None:
        _store = RealTimeFeatureStore()
    return _store
