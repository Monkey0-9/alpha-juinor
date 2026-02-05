"""
Order Book Pressure Analyzer - Microstructure Intelligence.

Features:
- Order imbalance detection
- Trade flow toxicity
- Price prediction from book
- Execution timing signals
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    size: int
    order_count: int


@dataclass
class OrderBook:
    """Full order book snapshot."""
    symbol: str
    timestamp: float
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0


@dataclass
class PressureSignal:
    """Order book pressure analysis result."""
    symbol: str
    imbalance_ratio: float  # -1 to 1 (negative = sell pressure)
    weighted_imbalance: float
    price_prediction: float  # Expected price move
    toxicity_score: float  # 0 to 1
    execution_signal: str  # "EXECUTE_NOW", "WAIT", "PASSIVE"
    confidence: float


class OrderBookAnalyzer:
    """
    Real-time order book analysis.

    Features:
    - Multi-level imbalance
    - Volume-weighted pressure
    - Trade flow analysis
    - Execution timing
    """

    def __init__(
        self,
        depth_levels: int = 5,
        imbalance_threshold: float = 0.3,
        toxicity_window: int = 100
    ):
        self.depth_levels = depth_levels
        self.imbalance_threshold = imbalance_threshold
        self.toxicity_window = toxicity_window

        # Book history
        self.book_history: Dict[str, deque] = {}

        # Trade history for toxicity
        self.trade_history: Dict[str, deque] = {}

    def update_book(self, book: OrderBook):
        """Update order book for a symbol."""
        symbol = book.symbol

        if symbol not in self.book_history:
            self.book_history[symbol] = deque(maxlen=100)

        self.book_history[symbol].append(book)

    def record_trade(
        self,
        symbol: str,
        price: float,
        size: int,
        side: str,  # "BUY" or "SELL"
        is_aggressive: bool
    ):
        """Record a trade for toxicity analysis."""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=self.toxicity_window)

        self.trade_history[symbol].append({
            "price": price,
            "size": size,
            "side": side,
            "aggressive": is_aggressive,
            "timestamp": time.time()
        })

    def calculate_imbalance(self, book: OrderBook) -> Tuple[float, float]:
        """
        Calculate order book imbalance.

        Returns:
            (simple_imbalance, weighted_imbalance)
        """
        bid_volume = sum(
            level.size for level in book.bids[:self.depth_levels]
        )
        ask_volume = sum(
            level.size for level in book.asks[:self.depth_levels]
        )

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0, 0.0

        simple = (bid_volume - ask_volume) / total

        # Weighted imbalance (closer levels matter more)
        weighted_bid = sum(
            level.size * (self.depth_levels - i)
            for i, level in enumerate(book.bids[:self.depth_levels])
        )
        weighted_ask = sum(
            level.size * (self.depth_levels - i)
            for i, level in enumerate(book.asks[:self.depth_levels])
        )

        weighted_total = weighted_bid + weighted_ask
        if weighted_total == 0:
            weighted = 0.0
        else:
            weighted = (weighted_bid - weighted_ask) / weighted_total

        return simple, weighted

    def predict_price_move(
        self,
        book: OrderBook,
        imbalance: float
    ) -> float:
        """
        Predict short-term price move from book imbalance.

        Based on research: imbalance is a leading indicator.
        """
        # Price sensitivity to imbalance
        # Higher imbalance = likely move in that direction
        sensitivity = book.spread * 0.5  # Half-spread per unit imbalance

        predicted_move = imbalance * sensitivity
        return predicted_move

    def calculate_toxicity(self, symbol: str) -> float:
        """
        Calculate trade flow toxicity.

        High toxicity = informed traders dominating flow.
        """
        if symbol not in self.trade_history:
            return 0.5

        trades = list(self.trade_history[symbol])

        if len(trades) < 10:
            return 0.5

        # Count aggressive trades
        aggressive_buy_vol = sum(
            t["size"] for t in trades
            if t["aggressive"] and t["side"] == "BUY"
        )
        aggressive_sell_vol = sum(
            t["size"] for t in trades
            if t["aggressive"] and t["side"] == "SELL"
        )

        total_vol = sum(t["size"] for t in trades)

        if total_vol == 0:
            return 0.5

        # Toxicity as absolute imbalance in aggressive flow
        toxicity = abs(aggressive_buy_vol - aggressive_sell_vol) / total_vol

        return min(1.0, toxicity)

    def analyze(self, book: OrderBook) -> PressureSignal:
        """Full order book analysis."""
        self.update_book(book)

        symbol = book.symbol

        # Calculate imbalance
        simple_imb, weighted_imb = self.calculate_imbalance(book)

        # Predict price move
        price_pred = self.predict_price_move(book, weighted_imb)

        # Calculate toxicity
        toxicity = self.calculate_toxicity(symbol)

        # Execution signal
        if abs(weighted_imb) > self.imbalance_threshold:
            if toxicity < 0.3:
                # Low toxicity, can execute
                execution = "EXECUTE_NOW"
                confidence = 0.8
            else:
                # High toxicity, wait
                execution = "WAIT"
                confidence = 0.6
        else:
            execution = "PASSIVE"
            confidence = 0.5

        return PressureSignal(
            symbol=symbol,
            imbalance_ratio=simple_imb,
            weighted_imbalance=weighted_imb,
            price_prediction=price_pred,
            toxicity_score=toxicity,
            execution_signal=execution,
            confidence=confidence
        )


# Global singleton
_book_analyzer: Optional[OrderBookAnalyzer] = None


def get_book_analyzer() -> OrderBookAnalyzer:
    """Get or create global order book analyzer."""
    global _book_analyzer
    if _book_analyzer is None:
        _book_analyzer = OrderBookAnalyzer()
    return _book_analyzer
