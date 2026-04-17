"""
Event-Driven Alpha - Earnings, M&A, Macro Events.

Strategies:
- Post-earnings announcement drift (PEAD)
- M&A arbitrage
- FOMC/macro event reactions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of tradeable events."""
    EARNINGS = "EARNINGS"
    MA_ANNOUNCEMENT = "MA_ANNOUNCEMENT"
    FOMC = "FOMC"
    CPI = "CPI"
    GDP = "GDP"
    DIVIDEND = "DIVIDEND"
    STOCK_SPLIT = "STOCK_SPLIT"


@dataclass
class EventSignal:
    """Event-driven trading signal."""
    symbol: str
    event_type: EventType
    event_date: str

    # Signal details
    direction: int  # 1 = long, -1 = short, 0 = neutral
    magnitude: float  # Expected move
    confidence: float

    # Timing
    days_to_event: int
    holding_period: int

    # Reasoning
    reasoning: str


@dataclass
class MarketEvent:
    """Market event data."""
    symbol: str
    event_type: EventType
    event_date: datetime
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    surprise: Optional[float] = None


class EventDrivenAlpha:
    """
    Event-driven strategy engine.

    Strategies:
    1. PEAD: Trade 1-5 days after earnings surprise
    2. M&A: Spread trades on announced deals
    3. Macro: Position before FOMC, CPI, etc.
    """

    def __init__(
        self,
        pead_holding_days: int = 5,
        surprise_threshold: float = 0.05,
        pre_event_days: int = 2
    ):
        self.pead_holding_days = pead_holding_days
        self.surprise_threshold = surprise_threshold
        self.pre_event_days = pre_event_days

        # Event calendar
        self.events: Dict[str, List[MarketEvent]] = {}

        # Historical event reactions
        self.reaction_history: List[Dict] = []

    def add_event(self, event: MarketEvent):
        """Add event to calendar."""
        if event.symbol not in self.events:
            self.events[event.symbol] = []
        self.events[event.symbol].append(event)

    def earnings_surprise_signal(
        self,
        symbol: str,
        expected_eps: float,
        actual_eps: float,
        historical_reactions: Optional[List[float]] = None
    ) -> EventSignal:
        """
        Generate PEAD (Post-Earnings Announcement Drift) signal.

        Strong positive surprise -> Long
        Strong negative surprise -> Short
        """
        surprise = (actual_eps - expected_eps) / abs(expected_eps) if expected_eps != 0 else 0

        if abs(surprise) < self.surprise_threshold:
            return EventSignal(
                symbol=symbol,
                event_type=EventType.EARNINGS,
                event_date=datetime.utcnow().strftime("%Y-%m-%d"),
                direction=0,
                magnitude=abs(surprise),
                confidence=0.0,
                days_to_event=0,
                holding_period=0,
                reasoning="Surprise below threshold"
            )

        # Direction based on surprise
        direction = 1 if surprise > 0 else -1

        # Confidence based on historical consistency
        if historical_reactions:
            # Check if historical reactions match current direction
            consistent = sum(1 for r in historical_reactions if r * direction > 0)
            consistency_rate = consistent / len(historical_reactions)
            confidence = min(1.0, abs(surprise) * 5 * consistency_rate)
        else:
            confidence = min(1.0, abs(surprise) * 5)

        return EventSignal(
            symbol=symbol,
            event_type=EventType.EARNINGS,
            event_date=datetime.utcnow().strftime("%Y-%m-%d"),
            direction=direction,
            magnitude=abs(surprise),
            confidence=confidence,
            days_to_event=0,
            holding_period=self.pead_holding_days,
            reasoning=f"EPS surprise: {surprise:.1%}"
        )

    def ma_arbitrage_signal(
        self,
        target_symbol: str,
        target_price: float,
        offer_price: float,
        deal_probability: float = 0.8,
        expected_close_days: int = 60
    ) -> EventSignal:
        """
        Generate M&A arbitrage signal.

        If deal spread > risk-adjusted threshold: Long target
        """
        spread = (offer_price - target_price) / target_price
        annualized_spread = spread * (365 / expected_close_days)

        # Risk-adjusted threshold (deal break probability)
        required_spread = (1 - deal_probability) * 0.10 + 0.05

        if annualized_spread > required_spread:
            signal = EventSignal(
                symbol=target_symbol,
                event_type=EventType.MA_ANNOUNCEMENT,
                event_date=datetime.utcnow().strftime("%Y-%m-%d"),
                direction=1,
                magnitude=spread,
                confidence=deal_probability,
                days_to_event=expected_close_days,
                holding_period=expected_close_days,
                reasoning=f"Deal spread: {spread:.1%}, prob: {deal_probability:.0%}"
            )
        else:
            signal = EventSignal(
                symbol=target_symbol,
                event_type=EventType.MA_ANNOUNCEMENT,
                event_date=datetime.utcnow().strftime("%Y-%m-%d"),
                direction=0,
                magnitude=spread,
                confidence=0.0,
                days_to_event=expected_close_days,
                holding_period=0,
                reasoning="Spread below threshold"
            )

        return signal

    def fomc_positioning_signal(
        self,
        days_to_fomc: int,
        market_expects_hike: bool,
        vix_level: float
    ) -> Tuple[str, float]:
        """
        Pre-FOMC positioning signal.

        Historical pattern: Markets often rally post-FOMC
        """
        if days_to_fomc > 5:
            return "NEUTRAL", 0.0

        # Pre-FOMC drift is typically positive
        # But reduce confidence in high VIX environments
        base_confidence = 0.6
        vix_adjustment = max(0, 1 - (vix_level - 20) * 0.05)

        confidence = base_confidence * vix_adjustment

        if days_to_fomc <= 2:
            return "LONG_SPY", confidence
        else:
            return "NEUTRAL", 0.0

    def get_upcoming_events(
        self,
        symbol: str,
        days_ahead: int = 7
    ) -> List[MarketEvent]:
        """Get upcoming events for a symbol."""
        if symbol not in self.events:
            return []

        cutoff = datetime.utcnow() + timedelta(days=days_ahead)
        upcoming = [
            e for e in self.events[symbol]
            if e.event_date <= cutoff
        ]

        return sorted(upcoming, key=lambda x: x.event_date)


# Global singleton
_event_driven: Optional[EventDrivenAlpha] = None


def get_event_driven_alpha() -> EventDrivenAlpha:
    """Get or create global event-driven alpha engine."""
    global _event_driven
    if _event_driven is None:
        _event_driven = EventDrivenAlpha()
    return _event_driven
