"""
Alternative Data Engine - SEC Filings and Sentiment.

Sources:
- SEC Form 4 (insider trades)
- SEC 13F (institutional holdings)
- News sentiment aggregation
- Analyst revisions
"""

import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class InsiderTrade:
    """Insider trading record from SEC Form 4."""
    symbol: str
    insider_name: str
    title: str
    transaction_type: str  # "BUY", "SELL", "GRANT"
    shares: float
    price: float
    date: str
    is_direct: bool


@dataclass
class InstitutionalHolding:
    """Institutional holding from SEC 13F."""
    symbol: str
    institution: str
    shares: float
    value: float
    pct_change: float  # Change from last quarter
    filing_date: str


@dataclass
class AlternativeDataSignal:
    """Combined alternative data signal."""
    symbol: str
    timestamp: str

    # Component signals
    insider_score: float  # -1 to 1 (sell to buy)
    institutional_score: float  # -1 to 1
    sentiment_score: float  # -1 to 1

    # Composite
    composite_score: float
    confidence: float

    # Details
    reasons: List[str] = field(default_factory=list)


class AlternativeDataEngine:
    """
    Aggregate alternative data sources for alpha signals.

    Note: Uses simulated data in demo mode.
    Production would integrate with SEC EDGAR, financial news APIs.
    """

    def __init__(
        self,
        insider_weight: float = 0.4,
        institutional_weight: float = 0.3,
        sentiment_weight: float = 0.3
    ):
        self.insider_weight = insider_weight
        self.institutional_weight = institutional_weight
        self.sentiment_weight = sentiment_weight

        # Cache for fetched data
        self.insider_cache: Dict[str, List[InsiderTrade]] = {}
        self.institutional_cache: Dict[str, List[InstitutionalHolding]] = {}
        self.sentiment_cache: Dict[str, float] = {}

    def get_insider_signal(self, symbol: str) -> float:
        """
        Get insider trading signal.

        Positive score = net insider buying
        Negative score = net insider selling

        Returns: -1 to 1
        """
        trades = self.insider_cache.get(symbol, [])

        if not trades:
            return 0.0

        # Calculate net insider sentiment (last 90 days)
        buy_value = sum(
            t.shares * t.price
            for t in trades
            if t.transaction_type == "BUY"
        )
        sell_value = sum(
            t.shares * t.price
            for t in trades
            if t.transaction_type == "SELL"
        )

        total = buy_value + sell_value
        if total == 0:
            return 0.0

        net = (buy_value - sell_value) / total
        return float(max(-1, min(1, net)))

    def get_institutional_signal(self, symbol: str) -> float:
        """
        Get institutional ownership change signal.

        Positive = institutions increasing positions
        Negative = institutions decreasing positions

        Returns: -1 to 1
        """
        holdings = self.institutional_cache.get(symbol, [])

        if not holdings:
            return 0.0

        # Average percent change across institutions
        changes = [h.pct_change for h in holdings if h.pct_change != 0]

        if not changes:
            return 0.0

        avg_change = sum(changes) / len(changes)

        # Normalize: 10% change = full signal
        normalized = avg_change / 0.10
        return float(max(-1, min(1, normalized)))

    def get_sentiment_signal(self, symbol: str) -> float:
        """
        Get news/social sentiment signal.

        Returns cached sentiment or 0 if not available.
        """
        return self.sentiment_cache.get(symbol, 0.0)

    def update_sentiment(self, symbol: str, score: float):
        """Update sentiment score for a symbol."""
        self.sentiment_cache[symbol] = max(-1, min(1, score))

    def add_insider_trade(self, trade: InsiderTrade):
        """Add an insider trade to the cache."""
        if trade.symbol not in self.insider_cache:
            self.insider_cache[trade.symbol] = []

        self.insider_cache[trade.symbol].append(trade)

        # Keep last 50 trades per symbol
        if len(self.insider_cache[trade.symbol]) > 50:
            self.insider_cache[trade.symbol] = self.insider_cache[trade.symbol][-50:]

    def add_institutional_holding(self, holding: InstitutionalHolding):
        """Add institutional holding to cache."""
        if holding.symbol not in self.institutional_cache:
            self.institutional_cache[holding.symbol] = []

        self.institutional_cache[holding.symbol].append(holding)

    def get_composite_signal(self, symbol: str) -> AlternativeDataSignal:
        """
        Get composite alternative data signal.
        """
        insider = self.get_insider_signal(symbol)
        institutional = self.get_institutional_signal(symbol)
        sentiment = self.get_sentiment_signal(symbol)

        # Weighted composite
        composite = (
            insider * self.insider_weight +
            institutional * self.institutional_weight +
            sentiment * self.sentiment_weight
        )

        # Confidence based on data availability
        data_points = sum([
            1 if symbol in self.insider_cache else 0,
            1 if symbol in self.institutional_cache else 0,
            1 if symbol in self.sentiment_cache else 0
        ])
        confidence = data_points / 3.0

        # Build reasons
        reasons = []
        if insider > 0.3:
            reasons.append("Strong insider buying")
        elif insider < -0.3:
            reasons.append("Significant insider selling")

        if institutional > 0.3:
            reasons.append("Institutions accumulating")
        elif institutional < -0.3:
            reasons.append("Institutions reducing positions")

        if sentiment > 0.3:
            reasons.append("Positive sentiment")
        elif sentiment < -0.3:
            reasons.append("Negative sentiment")

        return AlternativeDataSignal(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat(),
            insider_score=insider,
            institutional_score=institutional,
            sentiment_score=sentiment,
            composite_score=composite,
            confidence=confidence,
            reasons=reasons
        )

    def simulate_data(self, symbol: str):
        """
        Simulate alternative data for testing.

        In production, this would be replaced with real data fetches.
        """
        import random

        # Simulate insider trades
        for _ in range(random.randint(1, 5)):
            trade = InsiderTrade(
                symbol=symbol,
                insider_name=f"Insider_{random.randint(1, 10)}",
                title=random.choice(["CEO", "CFO", "Director", "VP"]),
                transaction_type=random.choice(["BUY", "SELL", "BUY", "BUY"]),  # Bias toward buy
                shares=random.randint(1000, 50000),
                price=random.uniform(50, 200),
                date=datetime.utcnow().isoformat(),
                is_direct=True
            )
            self.add_insider_trade(trade)

        # Simulate sentiment
        sentiment = random.uniform(-0.5, 0.5)
        self.update_sentiment(symbol, sentiment)


# Global singleton
_alt_data: Optional[AlternativeDataEngine] = None


def get_alternative_data_engine() -> AlternativeDataEngine:
    """Get or create global AlternativeDataEngine."""
    global _alt_data
    if _alt_data is None:
        _alt_data = AlternativeDataEngine()
    return _alt_data
