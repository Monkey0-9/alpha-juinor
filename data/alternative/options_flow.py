"""
Options Flow Analyzer - Real-Time Block Trade Detection
==========================================================

Elite-tier alternative data: analyze options flow for alpha.

Features:
1. Block trade detection (>$1M premium)
2. Unusual options activity
3. Put/Call ratio analysis
4. Smart money positioning
5. Options flow imbalance

Follow the flow. Follow the money.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
import threading

logger = logging.getLogger(__name__)


class OptionsFlowType(Enum):
    """Type of options flow."""
    CALL_SWEEP = "CALL_SWEEP"
    PUT_SWEEP = "PUT_SWEEP"
    CALL_BLOCK = "CALL_BLOCK"
    PUT_BLOCK = "PUT_BLOCK"
    CALL_SPLIT = "CALL_SPLIT"
    PUT_SPLIT = "PUT_SPLIT"
    UNUSUAL_CALL = "UNUSUAL_CALL"
    UNUSUAL_PUT = "UNUSUAL_PUT"


class FlowSentiment(Enum):
    """Sentiment from options flow."""
    STRONGLY_BULLISH = "STRONGLY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONGLY_BEARISH = "STRONGLY_BEARISH"


@dataclass
class OptionsFlow:
    """A single options flow event."""
    timestamp: datetime
    symbol: str

    # Contract details
    expiration: str
    strike: float
    option_type: str  # CALL, PUT

    # Trade details
    premium: float  # Total premium paid
    size: int  # Number of contracts
    price: float  # Price per contract

    # Flow type
    flow_type: OptionsFlowType

    # Analysis
    is_opening: bool  # Opening or closing position
    is_block: bool  # Block trade (>100 contracts)
    is_sweep: bool  # Sweep (hit multiple exchanges)
    at_ask: bool  # Buyer initiated

    # Calculated
    underlying_price: float
    days_to_expiry: int
    otm_pct: float  # % out of the money


@dataclass
class OptionsFlowSignal:
    """Aggregated options flow signal for a symbol."""
    timestamp: datetime
    symbol: str

    # Flow summary
    total_call_premium: float
    total_put_premium: float
    call_put_ratio: float

    # Block trades
    block_count_calls: int
    block_count_puts: int
    total_block_premium: float

    # Unusual activity
    unusual_flows: List[OptionsFlow]

    # Sentiment
    sentiment: FlowSentiment
    confidence: float

    # Signal
    signal: int  # 1 = bullish, -1 = bearish, 0 = neutral

    # Key insights
    insights: List[str]


class OptionsFlowAnalyzer:
    """
    Analyzes options flow for trading signals.

    Tracks:
    - Large block trades (>$1M premium)
    - Unusually high volume vs. open interest
    - Sweep orders across exchanges
    - Put/Call ratio deviations
    - Smart money positioning
    """

    # Thresholds
    BLOCK_PREMIUM_MIN = 100000  # $100k minimum for block
    LARGE_BLOCK_PREMIUM = 1000000  # $1M for large block
    UNUSUAL_VOLUME_MULTIPLIER = 3  # 3x normal volume
    PCR_BULLISH_THRESHOLD = 0.5  # Low PCR = bullish
    PCR_BEARISH_THRESHOLD = 1.5  # High PCR = bearish

    def __init__(self):
        """Initialize the analyzer."""
        self.flow_history: Dict[str, List[OptionsFlow]] = {}
        self.daily_stats: Dict[str, Dict] = {}
        self._lock = threading.Lock()

        logger.info(
            "[OPTIONS FLOW] Analyzer initialized - "
            "FOLLOWING THE MONEY"
        )

    def add_flow(self, flow: OptionsFlow):
        """Add a flow event to history."""
        with self._lock:
            if flow.symbol not in self.flow_history:
                self.flow_history[flow.symbol] = []

            self.flow_history[flow.symbol].append(flow)

            # Keep last 1000 flows per symbol
            if len(self.flow_history[flow.symbol]) > 1000:
                self.flow_history[flow.symbol] = self.flow_history[flow.symbol][-1000:]

            # Log significant flows
            if flow.premium >= self.LARGE_BLOCK_PREMIUM:
                logger.info(
                    f"[OPTIONS FLOW] LARGE BLOCK: {flow.symbol} "
                    f"{flow.option_type} ${flow.strike} {flow.expiration} | "
                    f"Premium: ${flow.premium:,.0f}"
                )

    def analyze_symbol(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Optional[OptionsFlowSignal]:
        """Analyze options flow for a symbol."""
        with self._lock:
            if symbol not in self.flow_history:
                return None

            flows = self.flow_history[symbol]

        if not flows:
            return None

        # Filter to lookback period
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        recent_flows = [f for f in flows if f.timestamp >= cutoff]

        if not recent_flows:
            return None

        # Aggregate metrics
        call_premium = sum(f.premium for f in recent_flows if f.option_type == "CALL")
        put_premium = sum(f.premium for f in recent_flows if f.option_type == "PUT")

        total_premium = call_premium + put_premium
        pcr = put_premium / call_premium if call_premium > 0 else 2.0

        # Block trades
        blocks_calls = [f for f in recent_flows if f.is_block and f.option_type == "CALL"]
        blocks_puts = [f for f in recent_flows if f.is_block and f.option_type == "PUT"]

        total_block_premium = sum(f.premium for f in blocks_calls + blocks_puts)

        # Unusual activity (sweeps + large blocks)
        unusual = [
            f for f in recent_flows
            if f.is_sweep or f.premium >= self.LARGE_BLOCK_PREMIUM
        ]

        # Determine sentiment
        sentiment, confidence = self._determine_sentiment(
            call_premium, put_premium, pcr,
            len(blocks_calls), len(blocks_puts),
            unusual
        )

        # Generate signal
        if sentiment == FlowSentiment.STRONGLY_BULLISH:
            signal = 1
        elif sentiment == FlowSentiment.BULLISH:
            signal = 1
        elif sentiment == FlowSentiment.STRONGLY_BEARISH:
            signal = -1
        elif sentiment == FlowSentiment.BEARISH:
            signal = -1
        else:
            signal = 0

        # Generate insights
        insights = self._generate_insights(
            symbol, call_premium, put_premium, pcr,
            blocks_calls, blocks_puts, unusual
        )

        return OptionsFlowSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            total_call_premium=call_premium,
            total_put_premium=put_premium,
            call_put_ratio=pcr,
            block_count_calls=len(blocks_calls),
            block_count_puts=len(blocks_puts),
            total_block_premium=total_block_premium,
            unusual_flows=unusual[:10],  # Top 10
            sentiment=sentiment,
            confidence=confidence,
            signal=signal,
            insights=insights
        )

    def _determine_sentiment(
        self,
        call_premium: float,
        put_premium: float,
        pcr: float,
        block_calls: int,
        block_puts: int,
        unusual: List[OptionsFlow]
    ) -> Tuple[FlowSentiment, float]:
        """Determine sentiment from flow data."""
        bullish_score = 0
        bearish_score = 0

        # Put/Call ratio
        if pcr < self.PCR_BULLISH_THRESHOLD:
            bullish_score += 2
        elif pcr > self.PCR_BEARISH_THRESHOLD:
            bearish_score += 2
        elif pcr < 1.0:
            bullish_score += 1
        else:
            bearish_score += 1

        # Block trade imbalance
        if block_calls > block_puts * 2:
            bullish_score += 2
        elif block_puts > block_calls * 2:
            bearish_score += 2
        elif block_calls > block_puts:
            bullish_score += 1
        elif block_puts > block_calls:
            bearish_score += 1

        # Unusual flow direction
        unusual_calls = [f for f in unusual if f.option_type == "CALL"]
        unusual_puts = [f for f in unusual if f.option_type == "PUT"]

        if len(unusual_calls) > len(unusual_puts) * 1.5:
            bullish_score += 2
        elif len(unusual_puts) > len(unusual_calls) * 1.5:
            bearish_score += 2

        # Premium imbalance
        if call_premium > put_premium * 2:
            bullish_score += 1
        elif put_premium > call_premium * 2:
            bearish_score += 1

        # Determine sentiment
        net_score = bullish_score - bearish_score
        total_score = bullish_score + bearish_score

        confidence = min(0.95, abs(net_score) / max(total_score, 1) + 0.5)

        if net_score >= 4:
            return FlowSentiment.STRONGLY_BULLISH, confidence
        elif net_score >= 2:
            return FlowSentiment.BULLISH, confidence
        elif net_score <= -4:
            return FlowSentiment.STRONGLY_BEARISH, confidence
        elif net_score <= -2:
            return FlowSentiment.BEARISH, confidence
        else:
            return FlowSentiment.NEUTRAL, 0.5

    def _generate_insights(
        self,
        symbol: str,
        call_premium: float,
        put_premium: float,
        pcr: float,
        block_calls: List[OptionsFlow],
        block_puts: List[OptionsFlow],
        unusual: List[OptionsFlow]
    ) -> List[str]:
        """Generate human-readable insights."""
        insights = []

        # Premium insight
        total = call_premium + put_premium
        if total >= 1000000:
            insights.append(
                f"High options activity: ${total/1e6:.1f}M total premium"
            )

        # PCR insight
        if pcr < 0.5:
            insights.append(
                f"Extremely bullish PCR: {pcr:.2f}"
            )
        elif pcr > 2.0:
            insights.append(
                f"Extremely bearish PCR: {pcr:.2f}"
            )

        # Block trade insight
        if block_calls:
            total_call_block = sum(f.premium for f in block_calls)
            if total_call_block >= 1000000:
                insights.append(
                    f"Large CALL block activity: ${total_call_block/1e6:.1f}M in {len(block_calls)} trades"
                )

        if block_puts:
            total_put_block = sum(f.premium for f in block_puts)
            if total_put_block >= 1000000:
                insights.append(
                    f"Large PUT block activity: ${total_put_block/1e6:.1f}M in {len(block_puts)} trades"
                )

        # Unusual activity
        if unusual:
            sweeps = [f for f in unusual if f.is_sweep]
            if sweeps:
                insights.append(
                    f"{len(sweeps)} sweep orders detected - aggressive execution"
                )

        return insights

    def simulate_flow_from_data(
        self,
        symbol: str,
        volume: float,
        price: float,
        returns_5d: float
    ) -> OptionsFlowSignal:
        """
        Simulate options flow signal from public data.

        Used when real options flow data is not available.
        """
        # Estimate premium based on volume and price
        estimated_call_premium = volume * price * 0.03 * (1 + max(0, returns_5d * 10))
        estimated_put_premium = volume * price * 0.03 * (1 - min(0, returns_5d * 10))

        pcr = estimated_put_premium / estimated_call_premium if estimated_call_premium > 0 else 1.0

        # Determine sentiment
        if returns_5d > 0.05 and pcr < 1.0:
            sentiment = FlowSentiment.BULLISH
            signal = 1
            confidence = 0.65
        elif returns_5d < -0.05 and pcr > 1.0:
            sentiment = FlowSentiment.BEARISH
            signal = -1
            confidence = 0.65
        else:
            sentiment = FlowSentiment.NEUTRAL
            signal = 0
            confidence = 0.5

        return OptionsFlowSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            total_call_premium=estimated_call_premium,
            total_put_premium=estimated_put_premium,
            call_put_ratio=pcr,
            block_count_calls=0,
            block_count_puts=0,
            total_block_premium=0,
            unusual_flows=[],
            sentiment=sentiment,
            confidence=confidence,
            signal=signal,
            insights=["Simulated from public data"]
        )

    def get_top_bullish(self, n: int = 10) -> List[Tuple[str, OptionsFlowSignal]]:
        """Get top N bullish symbols by flow."""
        signals = []

        with self._lock:
            for symbol in self.flow_history.keys():
                signal = self.analyze_symbol(symbol)
                if signal and signal.signal > 0:
                    signals.append((symbol, signal))

        # Sort by confidence * call premium
        signals.sort(
            key=lambda x: x[1].confidence * x[1].total_call_premium,
            reverse=True
        )

        return signals[:n]

    def get_top_bearish(self, n: int = 10) -> List[Tuple[str, OptionsFlowSignal]]:
        """Get top N bearish symbols by flow."""
        signals = []

        with self._lock:
            for symbol in self.flow_history.keys():
                signal = self.analyze_symbol(symbol)
                if signal and signal.signal < 0:
                    signals.append((symbol, signal))

        signals.sort(
            key=lambda x: x[1].confidence * x[1].total_put_premium,
            reverse=True
        )

        return signals[:n]


# Singleton
_analyzer: Optional[OptionsFlowAnalyzer] = None


def get_options_flow_analyzer() -> OptionsFlowAnalyzer:
    """Get or create the Options Flow Analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = OptionsFlowAnalyzer()
    return _analyzer
