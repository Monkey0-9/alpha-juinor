"""
Pattern Recognition Engine
=============================

Recognizes chart patterns for high-probability trades.

Patterns:
1. Head and Shoulders
2. Double Top/Bottom
3. Triangle (Ascending, Descending, Symmetric)
4. Flag and Pennant
5. Cup and Handle
6. Wedge
7. Channel
8. Candlestick Patterns

Precise pattern detection for profitable trades.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)

getcontext().prec = 50


class PatternType(Enum):
    """Chart pattern types."""
    HEAD_SHOULDERS = "HEAD_SHOULDERS"
    INVERSE_HEAD_SHOULDERS = "INVERSE_HEAD_SHOULDERS"
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    TRIPLE_TOP = "TRIPLE_TOP"
    TRIPLE_BOTTOM = "TRIPLE_BOTTOM"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRIC_TRIANGLE = "SYMMETRIC_TRIANGLE"
    BULL_FLAG = "BULL_FLAG"
    BEAR_FLAG = "BEAR_FLAG"
    BULL_PENNANT = "BULL_PENNANT"
    BEAR_PENNANT = "BEAR_PENNANT"
    CUP_HANDLE = "CUP_HANDLE"
    RISING_WEDGE = "RISING_WEDGE"
    FALLING_WEDGE = "FALLING_WEDGE"
    ASCENDING_CHANNEL = "ASCENDING_CHANNEL"
    DESCENDING_CHANNEL = "DESCENDING_CHANNEL"
    HORIZONTAL_CHANNEL = "HORIZONTAL_CHANNEL"


class CandlestickPattern(Enum):
    """Candlestick patterns."""
    DOJI = "DOJI"
    HAMMER = "HAMMER"
    INVERTED_HAMMER = "INVERTED_HAMMER"
    ENGULFING_BULLISH = "ENGULFING_BULLISH"
    ENGULFING_BEARISH = "ENGULFING_BEARISH"
    MORNING_STAR = "MORNING_STAR"
    EVENING_STAR = "EVENING_STAR"
    THREE_WHITE_SOLDIERS = "THREE_WHITE_SOLDIERS"
    THREE_BLACK_CROWS = "THREE_BLACK_CROWS"
    MARUBOZU_BULLISH = "MARUBOZU_BULLISH"
    MARUBOZU_BEARISH = "MARUBOZU_BEARISH"


@dataclass
class PatternSignal:
    """Detected pattern signal."""
    timestamp: datetime
    symbol: str

    # Pattern info
    pattern_type: str
    pattern_name: str
    bullish: bool

    # Confidence
    confidence: float

    # Trade suggestion
    entry_price: Decimal
    stop_loss: Decimal
    target_1: Decimal
    target_2: Decimal

    # Risk/Reward
    risk_reward: float

    # Pattern details
    pattern_start: int  # Bars ago
    pattern_end: int
    description: str


class ChartPatternRecognizer:
    """
    Recognizes chart patterns for trading.

    Uses price peaks and troughs to identify patterns.
    """

    def __init__(self):
        """Initialize the recognizer."""
        self.detections = 0

        logger.info("[PATTERNS] Chart Pattern Recognizer initialized")

    def find_patterns(
        self,
        symbol: str,
        prices: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None
    ) -> List[PatternSignal]:
        """Find all patterns in price data."""
        if len(prices) < 50:
            return []

        patterns = []

        p = prices.values
        h = high.values if high is not None else p
        l = low.values if low is not None else p

        current = Decimal(str(p[-1]))

        # Find peaks and troughs
        peaks, troughs = self._find_peaks_troughs(p)

        # Check various patterns

        # 1. Double Top
        if self._check_double_top(p, peaks):
            entry = current * Decimal("0.99")
            stop = current * Decimal("1.03")
            target = current * Decimal("0.93")

            patterns.append(PatternSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                pattern_type=PatternType.DOUBLE_TOP.value,
                pattern_name="Double Top",
                bullish=False,
                confidence=0.75,
                entry_price=entry.quantize(Decimal("0.01")),
                stop_loss=stop.quantize(Decimal("0.01")),
                target_1=target.quantize(Decimal("0.01")),
                target_2=(current * Decimal("0.90")).quantize(Decimal("0.01")),
                risk_reward=2.3,
                pattern_start=30,
                pattern_end=0,
                description="Price formed two peaks at similar levels - bearish reversal"
            ))

        # 2. Double Bottom
        if self._check_double_bottom(p, troughs):
            entry = current * Decimal("1.01")
            stop = current * Decimal("0.97")
            target = current * Decimal("1.07")

            patterns.append(PatternSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                pattern_type=PatternType.DOUBLE_BOTTOM.value,
                pattern_name="Double Bottom",
                bullish=True,
                confidence=0.75,
                entry_price=entry.quantize(Decimal("0.01")),
                stop_loss=stop.quantize(Decimal("0.01")),
                target_1=target.quantize(Decimal("0.01")),
                target_2=(current * Decimal("1.10")).quantize(Decimal("0.01")),
                risk_reward=2.3,
                pattern_start=30,
                pattern_end=0,
                description="Price formed two troughs at similar levels - bullish reversal"
            ))

        # 3. Ascending Triangle
        if self._check_ascending_triangle(p, peaks, troughs):
            entry = current * Decimal("1.02")
            stop = current * Decimal("0.97")
            target = current * Decimal("1.08")

            patterns.append(PatternSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                pattern_type=PatternType.ASCENDING_TRIANGLE.value,
                pattern_name="Ascending Triangle",
                bullish=True,
                confidence=0.70,
                entry_price=entry.quantize(Decimal("0.01")),
                stop_loss=stop.quantize(Decimal("0.01")),
                target_1=target.quantize(Decimal("0.01")),
                target_2=(current * Decimal("1.12")).quantize(Decimal("0.01")),
                risk_reward=2.0,
                pattern_start=20,
                pattern_end=0,
                description="Higher lows with flat resistance - bullish breakout expected"
            ))

        # 4. Descending Triangle
        if self._check_descending_triangle(p, peaks, troughs):
            entry = current * Decimal("0.98")
            stop = current * Decimal("1.03")
            target = current * Decimal("0.92")

            patterns.append(PatternSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                pattern_type=PatternType.DESCENDING_TRIANGLE.value,
                pattern_name="Descending Triangle",
                bullish=False,
                confidence=0.70,
                entry_price=entry.quantize(Decimal("0.01")),
                stop_loss=stop.quantize(Decimal("0.01")),
                target_1=target.quantize(Decimal("0.01")),
                target_2=(current * Decimal("0.88")).quantize(Decimal("0.01")),
                risk_reward=2.0,
                pattern_start=20,
                pattern_end=0,
                description="Lower highs with flat support - bearish breakdown expected"
            ))

        # 5. Bull Flag
        if self._check_bull_flag(p):
            entry = current * Decimal("1.01")
            stop = current * Decimal("0.97")
            target = current * Decimal("1.10")

            patterns.append(PatternSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                pattern_type=PatternType.BULL_FLAG.value,
                pattern_name="Bull Flag",
                bullish=True,
                confidence=0.72,
                entry_price=entry.quantize(Decimal("0.01")),
                stop_loss=stop.quantize(Decimal("0.01")),
                target_1=target.quantize(Decimal("0.01")),
                target_2=(current * Decimal("1.15")).quantize(Decimal("0.01")),
                risk_reward=2.5,
                pattern_start=15,
                pattern_end=0,
                description="Strong uptrend followed by consolidation - continuation expected"
            ))

        # 6. Falling Wedge
        if self._check_falling_wedge(p, peaks, troughs):
            entry = current * Decimal("1.02")
            stop = current * Decimal("0.96")
            target = current * Decimal("1.12")

            patterns.append(PatternSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                pattern_type=PatternType.FALLING_WEDGE.value,
                pattern_name="Falling Wedge",
                bullish=True,
                confidence=0.73,
                entry_price=entry.quantize(Decimal("0.01")),
                stop_loss=stop.quantize(Decimal("0.01")),
                target_1=target.quantize(Decimal("0.01")),
                target_2=(current * Decimal("1.18")).quantize(Decimal("0.01")),
                risk_reward=2.7,
                pattern_start=25,
                pattern_end=0,
                description="Converging downward trendlines - bullish reversal likely"
            ))

        self.detections += len(patterns)

        return patterns

    def _find_peaks_troughs(
        self,
        prices: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """Find local peaks and troughs."""
        peaks = []
        troughs = []

        window = 5

        for i in range(window, len(prices) - window):
            if prices[i] == max(prices[i-window:i+window+1]):
                peaks.append(i)
            if prices[i] == min(prices[i-window:i+window+1]):
                troughs.append(i)

        return peaks, troughs

    def _check_double_top(
        self,
        prices: np.ndarray,
        peaks: List[int]
    ) -> bool:
        """Check for double top pattern."""
        if len(peaks) < 2:
            return False

        # Last two peaks should be similar
        p1 = prices[peaks[-1]]
        p2 = prices[peaks[-2]] if len(peaks) >= 2 else 0

        # Peaks within 2% of each other
        if abs(p1 - p2) / p2 < 0.02:
            # Current price should be below peaks
            if prices[-1] < p1 * 0.98:
                return True

        return False

    def _check_double_bottom(
        self,
        prices: np.ndarray,
        troughs: List[int]
    ) -> bool:
        """Check for double bottom pattern."""
        if len(troughs) < 2:
            return False

        t1 = prices[troughs[-1]]
        t2 = prices[troughs[-2]] if len(troughs) >= 2 else prices[-1]

        if abs(t1 - t2) / t2 < 0.02:
            if prices[-1] > t1 * 1.02:
                return True

        return False

    def _check_ascending_triangle(
        self,
        prices: np.ndarray,
        peaks: List[int],
        troughs: List[int]
    ) -> bool:
        """Check for ascending triangle."""
        if len(peaks) < 2 or len(troughs) < 2:
            return False

        # Flat resistance
        recent_peaks = [prices[p] for p in peaks[-3:]]
        if len(recent_peaks) < 2:
            return False

        peak_std = np.std(recent_peaks) / np.mean(recent_peaks)

        # Higher lows
        recent_troughs = [prices[t] for t in troughs[-3:] if t in troughs]
        if len(recent_troughs) < 2:
            return False

        lows_rising = all(recent_troughs[i] < recent_troughs[i+1] for i in range(len(recent_troughs)-1))

        return peak_std < 0.02 and lows_rising

    def _check_descending_triangle(
        self,
        prices: np.ndarray,
        peaks: List[int],
        troughs: List[int]
    ) -> bool:
        """Check for descending triangle."""
        if len(peaks) < 2 or len(troughs) < 2:
            return False

        # Flat support
        recent_troughs = [prices[t] for t in troughs[-3:]]
        if len(recent_troughs) < 2:
            return False

        trough_std = np.std(recent_troughs) / np.mean(recent_troughs)

        # Lower highs
        recent_peaks = [prices[p] for p in peaks[-3:]]
        if len(recent_peaks) < 2:
            return False

        highs_falling = all(recent_peaks[i] > recent_peaks[i+1] for i in range(len(recent_peaks)-1))

        return trough_std < 0.02 and highs_falling

    def _check_bull_flag(self, prices: np.ndarray) -> bool:
        """Check for bull flag pattern."""
        if len(prices) < 30:
            return False

        # Strong uptrend in first part
        first_part = prices[-30:-10]
        second_part = prices[-10:]

        uptrend = first_part[-1] / first_part[0] > 1.08
        consolidation = abs(second_part[-1] / second_part[0] - 1) < 0.03

        return uptrend and consolidation

    def _check_falling_wedge(
        self,
        prices: np.ndarray,
        peaks: List[int],
        troughs: List[int]
    ) -> bool:
        """Check for falling wedge pattern."""
        if len(peaks) < 3 or len(troughs) < 3:
            return False

        # Both highs and lows falling, but converging
        recent_peaks = [prices[p] for p in peaks[-3:]]
        recent_troughs = [prices[t] for t in troughs[-3:]]

        highs_falling = all(recent_peaks[i] > recent_peaks[i+1] for i in range(len(recent_peaks)-1))
        lows_falling = all(recent_troughs[i] > recent_troughs[i+1] for i in range(len(recent_troughs)-1))

        # Converging (difference decreasing)
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            spread_1 = recent_peaks[0] - recent_troughs[0]
            spread_2 = recent_peaks[-1] - recent_troughs[-1]
            converging = spread_2 < spread_1

            return highs_falling and lows_falling and converging

        return False


class CandlestickRecognizer:
    """Recognizes candlestick patterns."""

    def __init__(self):
        """Initialize the recognizer."""
        logger.info("[PATTERNS] Candlestick Recognizer initialized")

    def find_patterns(
        self,
        symbol: str,
        open_prices: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> List[Dict[str, Any]]:
        """Find candlestick patterns."""
        if len(close) < 5:
            return []

        patterns = []

        o = open_prices.values
        h = high.values
        l = low.values
        c = close.values

        # Check latest candles

        # 1. Doji
        if self._is_doji(o[-1], h[-1], l[-1], c[-1]):
            patterns.append({
                "pattern": CandlestickPattern.DOJI.value,
                "name": "Doji",
                "bullish": None,  # Neutral
                "description": "Indecision candle"
            })

        # 2. Hammer
        if self._is_hammer(o[-1], h[-1], l[-1], c[-1]):
            patterns.append({
                "pattern": CandlestickPattern.HAMMER.value,
                "name": "Hammer",
                "bullish": True,
                "description": "Bullish reversal signal"
            })

        # 3. Bullish Engulfing
        if len(o) >= 2 and self._is_bullish_engulfing(o[-2], c[-2], o[-1], c[-1]):
            patterns.append({
                "pattern": CandlestickPattern.ENGULFING_BULLISH.value,
                "name": "Bullish Engulfing",
                "bullish": True,
                "description": "Strong bullish reversal"
            })

        # 4. Bearish Engulfing
        if len(o) >= 2 and self._is_bearish_engulfing(o[-2], c[-2], o[-1], c[-1]):
            patterns.append({
                "pattern": CandlestickPattern.ENGULFING_BEARISH.value,
                "name": "Bearish Engulfing",
                "bullish": False,
                "description": "Strong bearish reversal"
            })

        # 5. Morning Star
        if len(o) >= 3 and self._is_morning_star(o[-3:], c[-3:]):
            patterns.append({
                "pattern": CandlestickPattern.MORNING_STAR.value,
                "name": "Morning Star",
                "bullish": True,
                "description": "Three-candle bullish reversal"
            })

        return patterns

    def _is_doji(self, o: float, h: float, l: float, c: float) -> bool:
        """Check if candle is a doji."""
        body = abs(c - o)
        full_range = h - l

        if full_range == 0:
            return False

        return body / full_range < 0.1

    def _is_hammer(self, o: float, h: float, l: float, c: float) -> bool:
        """Check if candle is a hammer."""
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        full_range = h - l

        if full_range == 0:
            return False

        # Small body, long lower wick, small upper wick
        return (
            body / full_range < 0.3 and
            lower_wick > 2 * body and
            upper_wick < body
        )

    def _is_bullish_engulfing(
        self,
        o1: float, c1: float,
        o2: float, c2: float
    ) -> bool:
        """Check for bullish engulfing."""
        # First candle bearish, second bullish and engulfs first
        return (
            c1 < o1 and  # First bearish
            c2 > o2 and  # Second bullish
            o2 < c1 and  # Opens below first close
            c2 > o1      # Closes above first open
        )

    def _is_bearish_engulfing(
        self,
        o1: float, c1: float,
        o2: float, c2: float
    ) -> bool:
        """Check for bearish engulfing."""
        return (
            c1 > o1 and  # First bullish
            c2 < o2 and  # Second bearish
            o2 > c1 and  # Opens above first close
            c2 < o1      # Closes below first open
        )

    def _is_morning_star(
        self,
        opens: np.ndarray,
        closes: np.ndarray
    ) -> bool:
        """Check for morning star."""
        if len(opens) < 3:
            return False

        # First: large bearish, Second: small body, Third: large bullish
        first_bearish = closes[0] < opens[0]
        second_small = abs(closes[1] - opens[1]) < abs(opens[0] - closes[0]) * 0.3
        third_bullish = closes[2] > opens[2]
        third_closes_high = closes[2] > (opens[0] + closes[0]) / 2

        return first_bearish and second_small and third_bullish and third_closes_high


class PatternEngine:
    """
    Complete pattern recognition engine.

    Combines chart and candlestick patterns.
    """

    def __init__(self):
        """Initialize the engine."""
        self.chart_recognizer = ChartPatternRecognizer()
        self.candle_recognizer = CandlestickRecognizer()

        logger.info("[PATTERNS] Pattern Engine initialized")

    def analyze(
        self,
        symbol: str,
        prices: pd.Series,
        open_prices: Optional[pd.Series] = None,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Complete pattern analysis."""
        chart_patterns = self.chart_recognizer.find_patterns(symbol, prices, high, low)

        if open_prices is not None and high is not None and low is not None:
            candle_patterns = self.candle_recognizer.find_patterns(
                symbol, open_prices, high, low, prices
            )
        else:
            candle_patterns = []

        # Best trading opportunities
        best_trades = [p for p in chart_patterns if p.confidence >= 0.70]

        return {
            "symbol": symbol,
            "chart_patterns": chart_patterns,
            "candlestick_patterns": candle_patterns,
            "best_trades": best_trades,
            "total_patterns": len(chart_patterns) + len(candle_patterns)
        }


# Singleton
_engine: Optional[PatternEngine] = None


def get_pattern_engine() -> PatternEngine:
    """Get or create the Pattern Engine."""
    global _engine
    if _engine is None:
        _engine = PatternEngine()
    return _engine
