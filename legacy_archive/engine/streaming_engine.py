"""
Real-Time Streaming Engine - Sub-second Signal Generation.

Event-driven architecture for real-time trading:
- WebSocket price feed integration
- Async signal processing
- Low-latency execution trigger
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class MarketTick:
    """Real-time market data tick."""
    symbol: str
    price: float
    volume: float
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class StreamingSignal:
    """Real-time trading signal."""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    strength: float
    latency_ms: float
    timestamp: float


class TickBuffer:
    """Rolling buffer for recent ticks."""

    def __init__(self, max_size: int = 1000):
        self.buffer: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add(self, tick: MarketTick):
        with self.lock:
            self.buffer.append(tick)

    def get_recent(self, n: int = 100) -> List[MarketTick]:
        with self.lock:
            return list(self.buffer)[-n:]

    def get_vwap(self, n: int = 20) -> float:
        ticks = self.get_recent(n)
        if not ticks:
            return 0.0

        total_value = sum(t.price * t.volume for t in ticks)
        total_volume = sum(t.volume for t in ticks)

        return total_value / total_volume if total_volume > 0 else 0.0


class SignalProcessor:
    """
    Process ticks and generate signals.

    Fast signal generation for:
    - Momentum breakouts
    - VWAP crosses
    - Volume spikes
    """

    def __init__(self):
        self.buffers: Dict[str, TickBuffer] = {}
        self.last_signals: Dict[str, StreamingSignal] = {}
        self.signal_cooldown = 5.0  # Seconds between signals

    def get_buffer(self, symbol: str) -> TickBuffer:
        if symbol not in self.buffers:
            self.buffers[symbol] = TickBuffer()
        return self.buffers[symbol]

    def process_tick(self, tick: MarketTick) -> Optional[StreamingSignal]:
        """
        Process a tick and potentially generate a signal.
        """
        start_time = time.time()
        buffer = self.get_buffer(tick.symbol)
        buffer.add(tick)

        # Check cooldown
        last_signal = self.last_signals.get(tick.symbol)
        if last_signal and (tick.timestamp - last_signal.timestamp) < self.signal_cooldown:
            return None

        # Generate signal
        signal = self._generate_signal(tick.symbol, tick, buffer)

        if signal:
            latency_ms = (time.time() - start_time) * 1000
            signal.latency_ms = latency_ms
            self.last_signals[tick.symbol] = signal
            return signal

        return None

    def _generate_signal(
        self,
        symbol: str,
        tick: MarketTick,
        buffer: TickBuffer
    ) -> Optional[StreamingSignal]:
        """Generate signal from tick data."""
        ticks = buffer.get_recent(50)

        if len(ticks) < 20:
            return None

        # Calculate indicators
        prices = [t.price for t in ticks]
        volumes = [t.volume for t in ticks]

        current_price = prices[-1]
        vwap = buffer.get_vwap(20)
        avg_volume = sum(volumes[:-1]) / (len(volumes) - 1)
        current_volume = volumes[-1]

        # Short-term momentum
        momentum = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0

        # Volume spike
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Generate signal
        signal = None
        strength = 0.0

        # Momentum breakout with volume confirmation
        if momentum > 0.002 and volume_ratio > 1.5:
            signal = "BUY"
            strength = min(1.0, momentum * 100 * volume_ratio)
        elif momentum < -0.002 and volume_ratio > 1.5:
            signal = "SELL"
            strength = min(1.0, abs(momentum) * 100 * volume_ratio)

        # VWAP cross
        prev_prices = prices[-5:-1]
        if all(p < vwap for p in prev_prices) and current_price > vwap:
            signal = "BUY"
            strength = max(strength, 0.5)
        elif all(p > vwap for p in prev_prices) and current_price < vwap:
            signal = "SELL"
            strength = max(strength, 0.5)

        if signal:
            return StreamingSignal(
                symbol=symbol,
                signal=signal,
                strength=strength,
                latency_ms=0.0,
                timestamp=tick.timestamp
            )

        return None


class StreamingEngine:
    """
    Main streaming engine for real-time trading.

    Manages:
    - Tick ingestion
    - Signal processing
    - Execution triggering
    """

    def __init__(self):
        self.processor = SignalProcessor()
        self.callbacks: List[Callable[[StreamingSignal], None]] = []
        self.running = False
        self.tick_count = 0
        self.signal_count = 0

    def register_callback(self, callback: Callable[[StreamingSignal], None]):
        """Register callback for signals."""
        self.callbacks.append(callback)

    def on_tick(self, tick: MarketTick):
        """Process incoming tick."""
        self.tick_count += 1

        signal = self.processor.process_tick(tick)

        if signal:
            self.signal_count += 1
            logger.info(
                f"StreamingSignal: {signal.symbol} {signal.signal} "
                f"strength={signal.strength:.2f} latency={signal.latency_ms:.2f}ms"
            )

            for callback in self.callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    def simulate_tick(
        self,
        symbol: str,
        base_price: float,
        volatility: float = 0.001
    ) -> MarketTick:
        """Generate simulated tick for testing."""
        import random

        price_change = random.gauss(0, volatility)
        price = base_price * (1 + price_change)
        volume = random.randint(100, 10000)

        return MarketTick(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=time.time(),
            bid=price - 0.01,
            ask=price + 0.01
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "ticks_processed": self.tick_count,
            "signals_generated": self.signal_count,
            "signal_rate": self.signal_count / max(self.tick_count, 1),
            "symbols_tracked": len(self.processor.buffers)
        }


# Global singleton
_streaming_engine: Optional[StreamingEngine] = None


def get_streaming_engine() -> StreamingEngine:
    """Get or create global streaming engine."""
    global _streaming_engine
    if _streaming_engine is None:
        _streaming_engine = StreamingEngine()
    return _streaming_engine
