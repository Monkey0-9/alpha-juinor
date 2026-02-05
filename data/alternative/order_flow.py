"""
Order Flow Analyzer - Market Microstructure Signals
======================================================

Elite-tier alternative data: order flow imbalance detection.

Features:
1. Buy/Sell pressure detection
2. Lit vs. dark pool analysis
3. Large trade impact estimation
4. Order flow imbalance
5. Market microstructure signals

The order book tells the truth. Listen to it.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import threading

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Type of order."""
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"


class VenueType(Enum):
    """Type of venue."""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    ARCA = "ARCA"
    BATS = "BATS"
    IEX = "IEX"
    DARK_POOL = "DARK_POOL"
    ATS = "ATS"  # Alternative Trading System


class FlowDirection(Enum):
    """Direction of flow."""
    AGGRESSIVE_BUY = "AGGRESSIVE_BUY"
    AGGRESSIVE_SELL = "AGGRESSIVE_SELL"
    PASSIVE_BUY = "PASSIVE_BUY"
    PASSIVE_SELL = "PASSIVE_SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class OrderFlowTick:
    """A single order flow tick."""
    timestamp: datetime
    symbol: str

    # Trade details
    price: float
    size: int
    value: float

    # Order classification
    order_type: OrderType
    venue: VenueType
    is_dark: bool

    # Flow direction
    is_buyer_initiated: bool
    aggressor: str  # BUYER, SELLER

    # Calculated
    impact_pct: float  # Price impact %


@dataclass
class OrderFlowSignal:
    """Aggregated order flow signal."""
    timestamp: datetime
    symbol: str

    # Volume breakdown
    total_volume: int
    buy_volume: int
    sell_volume: int
    dark_volume: int
    lit_volume: int

    # Value breakdown
    total_value: float
    buy_value: float
    sell_value: float

    # Imbalance
    volume_imbalance: float  # -1 to 1
    value_imbalance: float  # -1 to 1
    net_pressure: float

    # Dark pool ratio
    dark_ratio: float

    # Flow metrics
    aggressive_buy_pct: float
    aggressive_sell_pct: float

    # Large trades
    large_buy_count: int
    large_sell_count: int
    large_trade_imbalance: int

    # Direction
    flow_direction: FlowDirection
    direction_confidence: float

    # Signal
    signal: int  # 1 = bullish, -1 = bearish, 0 = neutral

    # Insights
    insights: List[str]


class OrderFlowAnalyzer:
    """
    Analyzes order flow for trading signals.

    Detects:
    - Buy/sell pressure imbalance
    - Aggressive execution patterns
    - Dark pool activity
    - Large trade impact
    """

    # Thresholds
    LARGE_TRADE_SIZE = 10000  # 10k shares
    LARGE_TRADE_VALUE = 500000  # $500k
    SIGNIFICANT_IMBALANCE = 0.3  # 30% imbalance
    HIGH_DARK_RATIO = 0.4  # 40% dark

    def __init__(self):
        """Initialize the analyzer."""
        self.flow_history: Dict[str, List[OrderFlowTick]] = {}
        self._lock = threading.Lock()

        logger.info(
            "[ORDER FLOW] Analyzer initialized - "
            "LISTENING TO THE TAPE"
        )

    def add_tick(self, tick: OrderFlowTick):
        """Add an order flow tick."""
        with self._lock:
            if tick.symbol not in self.flow_history:
                self.flow_history[tick.symbol] = []

            self.flow_history[tick.symbol].append(tick)

            # Keep last 10000 ticks per symbol
            if len(self.flow_history[tick.symbol]) > 10000:
                self.flow_history[tick.symbol] = self.flow_history[tick.symbol][-10000:]

    def analyze_symbol(
        self,
        symbol: str,
        lookback_minutes: int = 60
    ) -> Optional[OrderFlowSignal]:
        """Analyze order flow for a symbol."""
        with self._lock:
            if symbol not in self.flow_history:
                return None

            ticks = self.flow_history[symbol]

        if not ticks:
            return None

        # Filter to lookback period
        cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)
        recent = [t for t in ticks if t.timestamp >= cutoff]

        if not recent:
            return self._neutral_signal(symbol)

        # Calculate volumes
        total_volume = sum(t.size for t in recent)
        buy_volume = sum(t.size for t in recent if t.is_buyer_initiated)
        sell_volume = total_volume - buy_volume

        dark_volume = sum(t.size for t in recent if t.is_dark)
        lit_volume = total_volume - dark_volume

        # Calculate values
        total_value = sum(t.value for t in recent)
        buy_value = sum(t.value for t in recent if t.is_buyer_initiated)
        sell_value = total_value - buy_value

        # Imbalances
        volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        value_imbalance = (buy_value - sell_value) / total_value if total_value > 0 else 0

        net_pressure = (volume_imbalance + value_imbalance) / 2

        # Dark ratio
        dark_ratio = dark_volume / total_volume if total_volume > 0 else 0

        # Aggressive flow
        aggressive_buys = [t for t in recent if t.aggressor == "BUYER" and t.order_type == OrderType.MARKET_BUY]
        aggressive_sells = [t for t in recent if t.aggressor == "SELLER" and t.order_type == OrderType.MARKET_SELL]

        aggressive_buy_pct = len(aggressive_buys) / len(recent) if recent else 0
        aggressive_sell_pct = len(aggressive_sells) / len(recent) if recent else 0

        # Large trades
        large_buys = [t for t in recent if t.is_buyer_initiated and (t.size >= self.LARGE_TRADE_SIZE or t.value >= self.LARGE_TRADE_VALUE)]
        large_sells = [t for t in recent if not t.is_buyer_initiated and (t.size >= self.LARGE_TRADE_SIZE or t.value >= self.LARGE_TRADE_VALUE)]

        large_trade_imbalance = len(large_buys) - len(large_sells)

        # Flow direction
        flow_direction, confidence = self._determine_direction(
            volume_imbalance, value_imbalance,
            aggressive_buy_pct, aggressive_sell_pct,
            large_trade_imbalance
        )

        # Signal
        if flow_direction == FlowDirection.AGGRESSIVE_BUY:
            signal = 1
        elif flow_direction == FlowDirection.PASSIVE_BUY and confidence > 0.6:
            signal = 1
        elif flow_direction == FlowDirection.AGGRESSIVE_SELL:
            signal = -1
        elif flow_direction == FlowDirection.PASSIVE_SELL and confidence > 0.6:
            signal = -1
        else:
            signal = 0

        # Insights
        insights = self._generate_insights(
            symbol, volume_imbalance, dark_ratio,
            large_buys, large_sells, flow_direction
        )

        return OrderFlowSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            total_volume=total_volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            dark_volume=dark_volume,
            lit_volume=lit_volume,
            total_value=total_value,
            buy_value=buy_value,
            sell_value=sell_value,
            volume_imbalance=volume_imbalance,
            value_imbalance=value_imbalance,
            net_pressure=net_pressure,
            dark_ratio=dark_ratio,
            aggressive_buy_pct=aggressive_buy_pct,
            aggressive_sell_pct=aggressive_sell_pct,
            large_buy_count=len(large_buys),
            large_sell_count=len(large_sells),
            large_trade_imbalance=large_trade_imbalance,
            flow_direction=flow_direction,
            direction_confidence=confidence,
            signal=signal,
            insights=insights
        )

    def _determine_direction(
        self,
        vol_imbalance: float,
        val_imbalance: float,
        agg_buy_pct: float,
        agg_sell_pct: float,
        large_imbalance: int
    ) -> Tuple[FlowDirection, float]:
        """Determine flow direction."""
        bullish_score = 0
        bearish_score = 0

        # Volume imbalance
        if vol_imbalance > self.SIGNIFICANT_IMBALANCE:
            bullish_score += 2
        elif vol_imbalance < -self.SIGNIFICANT_IMBALANCE:
            bearish_score += 2
        elif vol_imbalance > 0:
            bullish_score += 1
        else:
            bearish_score += 1

        # Value imbalance
        if val_imbalance > self.SIGNIFICANT_IMBALANCE:
            bullish_score += 2
        elif val_imbalance < -self.SIGNIFICANT_IMBALANCE:
            bearish_score += 2

        # Aggressive flow
        if agg_buy_pct > 0.4:
            bullish_score += 2
        elif agg_sell_pct > 0.4:
            bearish_score += 2

        # Large trade imbalance
        if large_imbalance >= 2:
            bullish_score += 2
        elif large_imbalance <= -2:
            bearish_score += 2
        elif large_imbalance > 0:
            bullish_score += 1
        elif large_imbalance < 0:
            bearish_score += 1

        # Determine direction
        net = bullish_score - bearish_score
        total = bullish_score + bearish_score

        confidence = min(0.95, abs(net) / max(total, 1) + 0.5)

        if net >= 4 and agg_buy_pct > agg_sell_pct:
            return FlowDirection.AGGRESSIVE_BUY, confidence
        elif net >= 2:
            return FlowDirection.PASSIVE_BUY, confidence
        elif net <= -4 and agg_sell_pct > agg_buy_pct:
            return FlowDirection.AGGRESSIVE_SELL, confidence
        elif net <= -2:
            return FlowDirection.PASSIVE_SELL, confidence
        else:
            return FlowDirection.NEUTRAL, 0.5

    def _generate_insights(
        self,
        symbol: str,
        vol_imbalance: float,
        dark_ratio: float,
        large_buys: List,
        large_sells: List,
        direction: FlowDirection
    ) -> List[str]:
        """Generate insights."""
        insights = []

        # Imbalance insight
        if abs(vol_imbalance) > self.SIGNIFICANT_IMBALANCE:
            direction_text = "buying" if vol_imbalance > 0 else "selling"
            insights.append(
                f"Significant {direction_text} pressure: {abs(vol_imbalance):.0%}"
            )

        # Dark pool insight
        if dark_ratio > self.HIGH_DARK_RATIO:
            insights.append(
                f"High dark pool activity: {dark_ratio:.0%}"
            )

        # Large trades
        if large_buys:
            total_val = sum(t.value for t in large_buys)
            insights.append(
                f"{len(large_buys)} large buys (${total_val/1e6:.1f}M)"
            )

        if large_sells:
            total_val = sum(t.value for t in large_sells)
            insights.append(
                f"{len(large_sells)} large sells (${total_val/1e6:.1f}M)"
            )

        # Direction insight
        if direction == FlowDirection.AGGRESSIVE_BUY:
            insights.append("AGGRESSIVE BUY PRESSURE detected")
        elif direction == FlowDirection.AGGRESSIVE_SELL:
            insights.append("AGGRESSIVE SELL PRESSURE detected")

        return insights

    def _neutral_signal(self, symbol: str) -> OrderFlowSignal:
        """Return neutral signal."""
        return OrderFlowSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            total_volume=0,
            buy_volume=0,
            sell_volume=0,
            dark_volume=0,
            lit_volume=0,
            total_value=0,
            buy_value=0,
            sell_value=0,
            volume_imbalance=0,
            value_imbalance=0,
            net_pressure=0,
            dark_ratio=0,
            aggressive_buy_pct=0,
            aggressive_sell_pct=0,
            large_buy_count=0,
            large_sell_count=0,
            large_trade_imbalance=0,
            flow_direction=FlowDirection.NEUTRAL,
            direction_confidence=0.5,
            signal=0,
            insights=["No recent order flow data"]
        )

    def simulate_from_candle(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int
    ) -> OrderFlowSignal:
        """
        Simulate order flow from OHLCV data.

        Uses price action to estimate buy/sell pressure.
        """
        # True range for volatility
        tr = high - low

        # Estimate buy/sell pressure from price action
        # Close > Open = more buying, Close < Open = more selling
        price_direction = 1 if close > open_price else -1

        # Position in range
        if tr > 0:
            close_position = (close - low) / tr  # 0-1
        else:
            close_position = 0.5

        # Volume split
        buy_pct = 0.5 + (close_position - 0.5) * 0.3  # 35-65% range

        buy_volume = int(volume * buy_pct)
        sell_volume = volume - buy_volume

        volume_imbalance = (buy_volume - sell_volume) / volume if volume > 0 else 0

        # Simulate value
        avg_price = (high + low + close) / 3
        total_value = volume * avg_price
        buy_value = buy_volume * avg_price
        sell_value = sell_volume * avg_price

        value_imbalance = volume_imbalance

        # Direction
        if volume_imbalance > 0.15 and close > open_price:
            direction = FlowDirection.AGGRESSIVE_BUY
            confidence = 0.65
            signal = 1
        elif volume_imbalance < -0.15 and close < open_price:
            direction = FlowDirection.AGGRESSIVE_SELL
            confidence = 0.65
            signal = -1
        elif volume_imbalance > 0:
            direction = FlowDirection.PASSIVE_BUY
            confidence = 0.55
            signal = 1 if volume_imbalance > 0.1 else 0
        elif volume_imbalance < 0:
            direction = FlowDirection.PASSIVE_SELL
            confidence = 0.55
            signal = -1 if volume_imbalance < -0.1 else 0
        else:
            direction = FlowDirection.NEUTRAL
            confidence = 0.5
            signal = 0

        return OrderFlowSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            total_volume=volume,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            dark_volume=0,
            lit_volume=volume,
            total_value=total_value,
            buy_value=buy_value,
            sell_value=sell_value,
            volume_imbalance=volume_imbalance,
            value_imbalance=value_imbalance,
            net_pressure=volume_imbalance,
            dark_ratio=0,
            aggressive_buy_pct=buy_pct if close > open_price else 0,
            aggressive_sell_pct=(1-buy_pct) if close < open_price else 0,
            large_buy_count=0,
            large_sell_count=0,
            large_trade_imbalance=0,
            flow_direction=direction,
            direction_confidence=confidence,
            signal=signal,
            insights=["Simulated from OHLCV"]
        )


# Singleton
_analyzer: Optional[OrderFlowAnalyzer] = None


def get_order_flow_analyzer() -> OrderFlowAnalyzer:
    """Get or create the Order Flow Analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = OrderFlowAnalyzer()
    return _analyzer
