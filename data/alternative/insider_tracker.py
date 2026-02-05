"""
Insider Trading Tracker - SEC Form 4 Parser
==============================================

Elite-tier alternative data: track insider activity.

Features:
1. SEC Form 4 filing parser
2. Insider sentiment aggregation
3. Cluster detection (multiple insiders buying)
4. Insider confidence scoring
5. Real-time filing alerts

Follow the insiders. They know more than us.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class InsiderRole(Enum):
    """Role of insider."""
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    CTO = "CTO"
    DIRECTOR = "DIRECTOR"
    VP = "VP"
    OFFICER = "OFFICER"
    TEN_PCT_OWNER = "10% OWNER"
    BENEFICIAL_OWNER = "BENEFICIAL OWNER"


class TransactionType(Enum):
    """Type of insider transaction."""
    PURCHASE = "PURCHASE"
    SALE = "SALE"
    OPTION_EXERCISE = "OPTION_EXERCISE"
    GIFT = "GIFT"
    ACQUISITION = "ACQUISITION"
    DISPOSITION = "DISPOSITION"


class InsiderSentiment(Enum):
    """Insider sentiment for a stock."""
    STRONGLY_BULLISH = "STRONGLY_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONGLY_BEARISH = "STRONGLY_BEARISH"


@dataclass
class InsiderTransaction:
    """A single insider transaction from SEC Form 4."""
    filing_date: datetime
    transaction_date: datetime
    symbol: str
    company_name: str

    # Insider info
    insider_name: str
    insider_role: InsiderRole
    relationship: str

    # Transaction details
    transaction_type: TransactionType
    shares: int
    price: float
    value: float  # shares * price

    # Ownership change
    shares_owned_before: int
    shares_owned_after: int
    ownership_change_pct: float

    # Direct or indirect
    is_direct: bool

    # Derived
    is_purchase: bool
    is_significant: bool  # >$100k or >10% of holdings


@dataclass
class InsiderSignal:
    """Aggregated insider signal for a symbol."""
    timestamp: datetime
    symbol: str

    # Recent activity
    transactions_30d: int
    purchases_30d: int
    sales_30d: int

    # Value
    total_purchase_value_30d: float
    total_sale_value_30d: float
    net_value_30d: float

    # Cluster detection
    unique_insiders_buying: int
    unique_insiders_selling: int
    cluster_detected: bool
    cluster_type: str  # BUYING_CLUSTER, SELLING_CLUSTER, NONE

    # Sentiment
    sentiment: InsiderSentiment
    confidence: float

    # Signal
    signal: int  # 1 = bullish, -1 = bearish, 0 = neutral

    # Key transactions
    key_transactions: List[InsiderTransaction]
    insights: List[str]


class InsiderTracker:
    """
    Tracks insider trading activity.

    Elite edge: Insiders know more than we do.
    Track their trades systematically.
    """

    # Thresholds
    SIGNIFICANT_VALUE = 100000  # $100k
    CLUSTER_THRESHOLD = 2  # 2+ insiders
    C_SUITE_WEIGHT = 3  # C-suite transactions count more

    def __init__(self):
        """Initialize the tracker."""
        self.transactions: Dict[str, List[InsiderTransaction]] = {}
        self._lock = threading.Lock()

        logger.info(
            "[INSIDER] Tracker initialized - "
            "FOLLOWING THE INSIDERS"
        )

    def add_transaction(self, tx: InsiderTransaction):
        """Add an insider transaction."""
        with self._lock:
            if tx.symbol not in self.transactions:
                self.transactions[tx.symbol] = []

            self.transactions[tx.symbol].append(tx)

            # Keep last 500 transactions per symbol
            if len(self.transactions[tx.symbol]) > 500:
                self.transactions[tx.symbol] = self.transactions[tx.symbol][-500:]

            # Log significant transactions
            if tx.is_significant and tx.is_purchase:
                logger.info(
                    f"[INSIDER] SIGNIFICANT PURCHASE: {tx.symbol} | "
                    f"{tx.insider_name} ({tx.insider_role.value}) | "
                    f"${tx.value:,.0f}"
                )

    def analyze_symbol(
        self,
        symbol: str,
        lookback_days: int = 30
    ) -> Optional[InsiderSignal]:
        """Analyze insider activity for a symbol."""
        with self._lock:
            if symbol not in self.transactions:
                return None

            all_tx = self.transactions[symbol]

        if not all_tx:
            return None

        # Filter to lookback period
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        recent = [tx for tx in all_tx if tx.filing_date >= cutoff]

        if not recent:
            return self._neutral_signal(symbol)

        # Aggregate
        purchases = [tx for tx in recent if tx.is_purchase]
        sales = [tx for tx in recent if not tx.is_purchase]

        total_purchase_value = sum(tx.value for tx in purchases)
        total_sale_value = sum(tx.value for tx in sales)
        net_value = total_purchase_value - total_sale_value

        # Unique insiders
        unique_buyers = len(set(tx.insider_name for tx in purchases))
        unique_sellers = len(set(tx.insider_name for tx in sales))

        # Cluster detection
        cluster_detected = False
        cluster_type = "NONE"

        if unique_buyers >= self.CLUSTER_THRESHOLD and unique_buyers > unique_sellers:
            cluster_detected = True
            cluster_type = "BUYING_CLUSTER"
        elif unique_sellers >= self.CLUSTER_THRESHOLD and unique_sellers > unique_buyers:
            cluster_detected = True
            cluster_type = "SELLING_CLUSTER"

        # Sentiment analysis
        sentiment, confidence = self._determine_sentiment(
            purchases, sales, net_value, unique_buyers, unique_sellers
        )

        # Signal
        if sentiment == InsiderSentiment.STRONGLY_BULLISH:
            signal = 1
        elif sentiment == InsiderSentiment.BULLISH:
            signal = 1
        elif sentiment == InsiderSentiment.STRONGLY_BEARISH:
            signal = -1
        elif sentiment == InsiderSentiment.BEARISH:
            signal = -1
        else:
            signal = 0

        # Key transactions
        key_tx = sorted(recent, key=lambda x: x.value, reverse=True)[:5]

        # Insights
        insights = self._generate_insights(
            symbol, purchases, sales, net_value, cluster_type, unique_buyers, unique_sellers
        )

        return InsiderSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            transactions_30d=len(recent),
            purchases_30d=len(purchases),
            sales_30d=len(sales),
            total_purchase_value_30d=total_purchase_value,
            total_sale_value_30d=total_sale_value,
            net_value_30d=net_value,
            unique_insiders_buying=unique_buyers,
            unique_insiders_selling=unique_sellers,
            cluster_detected=cluster_detected,
            cluster_type=cluster_type,
            sentiment=sentiment,
            confidence=confidence,
            signal=signal,
            key_transactions=key_tx,
            insights=insights
        )

    def _determine_sentiment(
        self,
        purchases: List[InsiderTransaction],
        sales: List[InsiderTransaction],
        net_value: float,
        unique_buyers: int,
        unique_sellers: int
    ) -> Tuple[InsiderSentiment, float]:
        """Determine insider sentiment."""
        bullish_score = 0
        bearish_score = 0

        # Net value direction
        if net_value > 1000000:
            bullish_score += 3
        elif net_value > 100000:
            bullish_score += 2
        elif net_value > 0:
            bullish_score += 1
        elif net_value < -1000000:
            bearish_score += 3
        elif net_value < -100000:
            bearish_score += 2
        elif net_value < 0:
            bearish_score += 1

        # C-suite activity
        csuite_roles = {InsiderRole.CEO, InsiderRole.CFO, InsiderRole.COO, InsiderRole.CTO}

        csuite_buyers = sum(1 for tx in purchases if tx.insider_role in csuite_roles)
        csuite_sellers = sum(1 for tx in sales if tx.insider_role in csuite_roles)

        if csuite_buyers > 0:
            bullish_score += 2 * csuite_buyers
        if csuite_sellers > 0:
            bearish_score += 2 * csuite_sellers

        # Cluster bonus
        if unique_buyers >= 3:
            bullish_score += 2
        if unique_sellers >= 3:
            bearish_score += 2

        # Transaction count
        if len(purchases) > len(sales) * 2:
            bullish_score += 1
        elif len(sales) > len(purchases) * 2:
            bearish_score += 1

        # Determine sentiment
        net_score = bullish_score - bearish_score
        total = bullish_score + bearish_score

        confidence = min(0.95, abs(net_score) / max(total, 1) + 0.5)

        if net_score >= 5:
            return InsiderSentiment.STRONGLY_BULLISH, confidence
        elif net_score >= 2:
            return InsiderSentiment.BULLISH, confidence
        elif net_score <= -5:
            return InsiderSentiment.STRONGLY_BEARISH, confidence
        elif net_score <= -2:
            return InsiderSentiment.BEARISH, confidence
        else:
            return InsiderSentiment.NEUTRAL, 0.5

    def _generate_insights(
        self,
        symbol: str,
        purchases: List[InsiderTransaction],
        sales: List[InsiderTransaction],
        net_value: float,
        cluster_type: str,
        unique_buyers: int,
        unique_sellers: int
    ) -> List[str]:
        """Generate insights from insider activity."""
        insights = []

        # Net value insight
        if abs(net_value) >= 1000000:
            direction = "buying" if net_value > 0 else "selling"
            insights.append(
                f"Net insider {direction}: ${abs(net_value)/1e6:.1f}M"
            )

        # Cluster insight
        if cluster_type == "BUYING_CLUSTER":
            insights.append(
                f"BUYING CLUSTER: {unique_buyers} insiders purchasing"
            )
        elif cluster_type == "SELLING_CLUSTER":
            insights.append(
                f"SELLING CLUSTER: {unique_sellers} insiders selling"
            )

        # C-suite insight
        csuite = [InsiderRole.CEO, InsiderRole.CFO, InsiderRole.COO]
        csuite_purchases = [tx for tx in purchases if tx.insider_role in csuite]

        if csuite_purchases:
            roles = set(tx.insider_role.value for tx in csuite_purchases)
            insights.append(
                f"C-suite buying: {', '.join(roles)}"
            )

        # Large transaction insight
        large_purchases = [tx for tx in purchases if tx.value >= 500000]
        if large_purchases:
            insights.append(
                f"{len(large_purchases)} purchases over $500k"
            )

        return insights

    def _neutral_signal(self, symbol: str) -> InsiderSignal:
        """Return neutral signal when no data."""
        return InsiderSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            transactions_30d=0,
            purchases_30d=0,
            sales_30d=0,
            total_purchase_value_30d=0,
            total_sale_value_30d=0,
            net_value_30d=0,
            unique_insiders_buying=0,
            unique_insiders_selling=0,
            cluster_detected=False,
            cluster_type="NONE",
            sentiment=InsiderSentiment.NEUTRAL,
            confidence=0.5,
            signal=0,
            key_transactions=[],
            insights=["No recent insider activity"]
        )

    def get_top_insider_buys(self, n: int = 10) -> List[Tuple[str, InsiderSignal]]:
        """Get top stocks by insider buying."""
        signals = []

        with self._lock:
            for symbol in self.transactions.keys():
                signal = self.analyze_symbol(symbol)
                if signal and signal.signal > 0:
                    signals.append((symbol, signal))

        signals.sort(
            key=lambda x: x[1].total_purchase_value_30d,
            reverse=True
        )

        return signals[:n]


# Singleton
_tracker: Optional[InsiderTracker] = None


def get_insider_tracker() -> InsiderTracker:
    """Get or create the Insider Tracker."""
    global _tracker
    if _tracker is None:
        _tracker = InsiderTracker()
    return _tracker
