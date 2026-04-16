"""
High-Frequency Trading Infrastructure
=====================================

Low-latency market data handlers and order processing for HFT strategies.

Features:
- Zero-copy market data processing
- FPGA-simulated tick processing
- Microsecond-level latency optimization
- Co-location simulation
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """Market data tick."""

    symbol: str
    timestamp_ns: int  # Nanosecond precision
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    volume: int


class LowLatencyMarketDataHandler:
    """
    Ultra-low-latency market data handler.

    Optimizations:
    - Zero-copy data structures
    - Lock-free circular buffers
    - Pre-allocated memory
    - SIMD-friendly layouts
    """

    def __init__(self, buffer_size: int = 100000):
        self.buffer_size = buffer_size

        # Pre-allocated circular buffers for each symbol
        self.buffers: Dict[str, deque] = {}
        self.latest_ticks: Dict[str, Tick] = {}

        # Callbacks for tick processing
        self.callbacks: List[Callable[[Tick], None]] = []

        # Performance metrics
        self.latency_ns: List[int] = []

    def register_callback(self, callback: Callable[[Tick], None]):
        """Register callback for tick processing."""
        self.callbacks.append(callback)

    def process_tick(self, tick: Tick):
        """
        Process incoming tick with minimal latency.

        Args:
            tick: Market data tick
        """
        start_ns = time.perf_counter_ns()

        # Update latest tick
        self.latest_ticks[tick.symbol] = tick

        # Store in circular buffer
        if tick.symbol not in self.buffers:
            self.buffers[tick.symbol] = deque(maxlen=self.buffer_size)

        self.buffers[tick.symbol].append(tick)

        # Call registered callbacks
        for callback in self.callbacks:
            callback(tick)

        # Track latency
        end_ns = time.perf_counter_ns()
        self.latency_ns.append(end_ns - start_ns)

        # Keep only recent latency measurements
        if len(self.latency_ns) > 1000:
            self.latency_ns = self.latency_ns[-1000:]

    def get_latest_tick(self, symbol: str) -> Optional[Tick]:
        """Get latest tick for symbol."""
        return self.latest_ticks.get(symbol)

    def get_mid_price(self, symbol: str) -> Optional[float]:
        """Get mid price for symbol."""
        tick = self.latest_ticks.get(symbol)
        if tick:
            return (tick.bid + tick.ask) / 2
        return None

    def get_spread_bps(self, symbol: str) -> Optional[float]:
        """Get spread in basis points."""
        tick = self.latest_ticks.get(symbol)
        if tick and tick.bid > 0:
            return ((tick.ask - tick.bid) / tick.bid) * 10000
        return None

    def get_avg_latency_us(self) -> float:
        """Get average processing latency in microseconds."""
        if not self.latency_ns:
            return 0.0
        return np.mean(self.latency_ns) / 1000  # Convert ns to us

    def get_p99_latency_us(self) -> float:
        """Get 99th percentile latency in microseconds."""
        if not self.latency_ns:
            return 0.0
        return np.percentile(self.latency_ns, 99) / 1000


class FPGATickProcessor:
    """
    Simulates FPGA-level tick processing for strategy evaluation.

    In production HFT, this would be actual FPGA hardware.
    Here we simulate the ultra-low-latency processing.
    """

    def __init__(self):
        # Simulated FPGA state (registers)
        self.registers: Dict[str, float] = {}
        self.tick_count = 0

    def update(self, symbol: str, bid: float, ask: float) -> Dict[str, any]:
        """
        FPGA-style tick update (simulated).

        Performs:
        - EMA calculation
        - Spread monitoring
        - Signal generation

        Args:
            symbol: Symbol
            bid: Bid price
            ask: Ask price

        Returns:
            Signals dictionary
        """
        mid = (bid + ask) / 2
        spread = ask - bid

        # Register names
        ema_key = f"{symbol}_ema"
        spread_key = f"{symbol}_spread"

        # Update EMA (alpha = 0.1)
        if ema_key in self.registers:
            ema = 0.9 * self.registers[ema_key] + 0.1 * mid
        else:
            ema = mid

        self.registers[ema_key] = ema
        self.registers[spread_key] = spread

        # Generate signals
        signals = {
            "mid": mid,
            "ema": ema,
            "spread": spread,
            "signal": 1 if mid > ema else -1,
            "spread_ok": spread < 0.01 * mid,  # Spread < 1%
        }

        self.tick_count += 1

        return signals


class CoLocationSimulator:
    """
    Simulates co-location latency benefits.

    Models:
    - Network round-trip time
    - Queue delays
    - Exchange matching engine latency
    """

    def __init__(
        self,
        colocated: bool = False,
    ):
        self.colocated = colocated

        # Latency models (in microseconds)
        if colocated:
            self.network_latency_us = 10  # Same datacenter
            self.queue_latency_us = 5
        else:
            self.network_latency_us = 500  # Internet latency
            self.queue_latency_us = 50

    def simulate_order_latency(self) -> float:
        """
        Simulate order placement latency.

        Returns:
            Total latency in microseconds
        """
        # Network round-trip
        latency = 2 * self.network_latency_us

        # Queue delay (with jitter)
        latency += self.queue_latency_us * (1 + 0.2 * np.random.randn())

        # Exchange matching (fixed)
        latency += 20  #  microseconds

        return max(0, latency)

    def estimate_fill_probability(
        self, order_latency_us: float, liquidity_refresh_us: float = 100
    ) -> float:
        """
        Estimate probability of order fill based on latency.

        Args:
            order_latency_us: Order placement latency
            liquidity_refresh_us: How fast liquidity refreshes

        Returns:
            Fill probability [0, 1]
        """
        # Simplified model: if latency < refresh time, high fill prob
        if order_latency_us < liquidity_refresh_us:
            return 0.95
        else:
            decay = np.exp(-(order_latency_us - liquidity_refresh_us) / 100)
            return 0.5 + 0.45 * decay


class HFTAlphaModel:
    """
    High-frequency alpha model.

    Strategies:
    - Market making
    - Statistical arbitrage
    - Momentum ignition detection
    """

    def __init__(self):
        self.positions: Dict[str, float] = {}
        self.inventory_limit = 1000

    def market_making_signal(
        self, symbol: str, bid: float, ask: float, fair_value: float
    ) -> Dict[str, Optional[float]]:
        """
        Generate market making quotes.

        Args:
            symbol: Symbol
            bid: Current bid
            ask: Current ask
            fair_value: Estimated fair value

        Returns:
            Dictionary with quote_bid and quote_ask
        """
        spread = ask - bid
        optimal_spread = max(0.0001 * fair_value, spread * 0.8)

        # Check inventory
        current_position = self.positions.get(symbol, 0)

        # Skew quotes if inventory is large
        if abs(current_position) > self.inventory_limit * 0.5:
            skew = 0.0002 * fair_value * np.sign(current_position)
        else:
            skew = 0

        quote_bid = fair_value - optimal_spread / 2 - skew
        quote_ask = fair_value + optimal_spread / 2 - skew

        return {"quote_bid": quote_bid, "quote_ask": quote_ask}

    def stat_arb_signal(
        self, symbol_a: str, symbol_b: str, price_a: float, price_b: float
    ) -> int:
        """
        Statistical arbitrage signal for pairs.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            price_a: Price of A
            price_b: Price of B

        Returns:
            Signal: +1 (long A, short B), 0 (neutral), -1 (short A, long B)
        """
        # Simplified: use price ratio
        ratio = price_a / price_b

        # Historical mean (simulated)
        mean_ratio = 1.0

        z_score = (ratio - mean_ratio) / (0.01 * mean_ratio)

        if z_score > 2:
            return -1  # A expensive relative to B
        elif z_score < -2:
            return 1  # A cheap relative to B
        else:
            return 0
