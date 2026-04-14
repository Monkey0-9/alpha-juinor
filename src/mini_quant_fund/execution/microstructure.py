"""
Market Microstructure Analysis.

Analyzes order book dynamics, trade flow, and execution quality.
Based on microstructure academic research.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OrderFlowDirection(Enum):
    """Direction of order flow pressure."""
    BUYER_INITIATED = "BUYER"
    SELLER_INITIATED = "SELLER"
    NEUTRAL = "NEUTRAL"


@dataclass
class MicrostructureSignal:
    """Microstructure-based trading signal."""
    symbol: str
    timestamp: str

    # Order book imbalance (-1 to 1, positive = buy pressure)
    book_imbalance: float

    # Trade flow imbalance
    flow_direction: OrderFlowDirection
    flow_intensity: float  # 0 to 1

    # Toxicity measures
    vpin: float  # Volume-synchronized probability of informed trading

    # Signal
    signal: float  # -1 to 1
    confidence: float  # 0 to 1


class MicrostructureAnalyzer:
    """
    Analyze market microstructure for execution intelligence.

    Features:
    - Order book imbalance
    - Trade flow classification (Lee-Ready)
    - VPIN (toxicity measure)
    - Spread dynamics
    """

    def __init__(
        self,
        vpin_buckets: int = 50,
        lookback_trades: int = 100
    ):
        self.vpin_buckets = vpin_buckets
        self.lookback_trades = lookback_trades

        # Cache for VPIN calculation
        self.trade_history: Dict[str, List[Dict]] = {}

    def analyze(
        self,
        symbol: str,
        trades: pd.DataFrame,
        quotes: Optional[pd.DataFrame] = None,
        timestamp: Optional[str] = None
    ) -> MicrostructureSignal:
        """
        Analyze microstructure for a symbol.

        Args:
            trades: DataFrame with columns [price, volume, timestamp]
            quotes: Optional DataFrame with [bid, ask, bid_size, ask_size]

        Returns:
            MicrostructureSignal
        """
        timestamp = timestamp or str(pd.Timestamp.utcnow())

        # Order book imbalance
        book_imbalance = self._calc_book_imbalance(quotes) if quotes is not None else 0

        # Trade flow classification
        flow_dir, flow_intensity = self._classify_trade_flow(trades)

        # VPIN
        vpin = self._calc_vpin(symbol, trades)

        # Combine into signal
        signal, confidence = self._generate_signal(
            book_imbalance, flow_dir, flow_intensity, vpin
        )

        return MicrostructureSignal(
            symbol=symbol,
            timestamp=timestamp,
            book_imbalance=book_imbalance,
            flow_direction=flow_dir,
            flow_intensity=flow_intensity,
            vpin=vpin,
            signal=signal,
            confidence=confidence
        )

    def _calc_book_imbalance(self, quotes: pd.DataFrame) -> float:
        """
        Calculate order book imbalance.

        Imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        Range: -1 (all ask) to 1 (all bid)
        """
        if quotes is None or quotes.empty:
            return 0.0

        latest = quotes.iloc[-1]
        bid_size = latest.get("bid_size", 0)
        ask_size = latest.get("ask_size", 0)

        total = bid_size + ask_size
        if total == 0:
            return 0.0

        return float((bid_size - ask_size) / total)

    def _classify_trade_flow(
        self, trades: pd.DataFrame
    ) -> Tuple[OrderFlowDirection, float]:
        """
        Classify trades as buyer or seller initiated (Lee-Ready).

        Uses tick rule:
        - If trade price > previous price: buyer initiated
        - If trade price < previous price: seller initiated
        """
        if trades is None or len(trades) < 2:
            return OrderFlowDirection.NEUTRAL, 0.0

        recent = trades.iloc[-self.lookback_trades:]

        if len(recent) < 2:
            return OrderFlowDirection.NEUTRAL, 0.0

        prices = recent["price"].values
        volumes = recent["volume"].values

        # Classify each trade
        buyer_vol = 0.0
        seller_vol = 0.0

        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                buyer_vol += volumes[i]
            elif prices[i] < prices[i-1]:
                seller_vol += volumes[i]

        total_vol = buyer_vol + seller_vol
        if total_vol == 0:
            return OrderFlowDirection.NEUTRAL, 0.0

        net_flow = (buyer_vol - seller_vol) / total_vol
        intensity = abs(net_flow)

        if net_flow > 0.1:
            return OrderFlowDirection.BUYER_INITIATED, intensity
        elif net_flow < -0.1:
            return OrderFlowDirection.SELLER_INITIATED, intensity
        else:
            return OrderFlowDirection.NEUTRAL, intensity

    def _calc_vpin(self, symbol: str, trades: pd.DataFrame) -> float:
        """
        Calculate VPIN (Volume-synchronized Probability of Informed Trading).

        Higher VPIN = more likely informed trading = higher toxicity.
        Range: 0 to 1
        """
        if trades is None or len(trades) < 50:
            return 0.5  # Default neutral

        prices = trades["price"].values
        volumes = trades["volume"].values

        # Volume buckets
        total_vol = volumes.sum()
        bucket_size = total_vol / self.vpin_buckets

        if bucket_size == 0:
            return 0.5

        # Classify trades in each bucket
        current_bucket_vol = 0
        buy_vol = 0
        sell_vol = 0
        bucket_imbalances = []

        for i in range(1, len(prices)):
            vol = volumes[i]

            # Tick rule classification
            if prices[i] > prices[i-1]:
                buy_vol += vol
            elif prices[i] < prices[i-1]:
                sell_vol += vol
            else:
                # Split evenly for unchanged price
                buy_vol += vol / 2
                sell_vol += vol / 2

            current_bucket_vol += vol

            if current_bucket_vol >= bucket_size:
                # Complete bucket
                total = buy_vol + sell_vol
                if total > 0:
                    imbalance = abs(buy_vol - sell_vol) / total
                    bucket_imbalances.append(imbalance)

                # Reset for next bucket
                current_bucket_vol = 0
                buy_vol = 0
                sell_vol = 0

        if not bucket_imbalances:
            return 0.5

        # VPIN = average of bucket imbalances
        vpin = np.mean(bucket_imbalances)
        return float(np.clip(vpin, 0, 1))

    def _generate_signal(
        self,
        book_imbalance: float,
        flow_dir: OrderFlowDirection,
        flow_intensity: float,
        vpin: float
    ) -> Tuple[float, float]:
        """
        Generate composite microstructure signal.

        Returns:
            (signal, confidence)
        """
        # Flow direction to numeric
        flow_numeric = {
            OrderFlowDirection.BUYER_INITIATED: 1.0,
            OrderFlowDirection.SELLER_INITIATED: -1.0,
            OrderFlowDirection.NEUTRAL: 0.0
        }[flow_dir]

        # Combine signals
        # - Book imbalance: immediate pressure
        # - Flow direction: recent trading activity
        # - VPIN: if high, informed traders are active (be cautious)

        raw_signal = (
            0.4 * book_imbalance +
            0.4 * flow_numeric * flow_intensity +
            0.2 * (0.5 - vpin)  # Lower VPIN = safer to trade
        )

        signal = float(np.clip(raw_signal, -1, 1))

        # Confidence based on data quality and VPIN
        # High VPIN = lower confidence (informed trading risk)
        confidence = float((1 - vpin) * flow_intensity)
        confidence = np.clip(confidence, 0, 1)

        return signal, confidence

    def get_execution_recommendation(
        self,
        signal: MicrostructureSignal
    ) -> Dict[str, str]:
        """
        Get execution recommendation based on microstructure.
        """
        recommendations = {}

        # High VPIN = toxic flow, be careful
        if signal.vpin > 0.7:
            recommendations["toxicity"] = "HIGH - Consider delaying execution"
            recommendations["algo"] = "VWAP_PASSIVE"
        elif signal.vpin > 0.5:
            recommendations["toxicity"] = "MEDIUM - Normal execution"
            recommendations["algo"] = "VWAP"
        else:
            recommendations["toxicity"] = "LOW - Safe to execute"
            recommendations["algo"] = "MARKET" if signal.flow_intensity > 0.5 else "LIMIT"

        # Order book imbalance
        if abs(signal.book_imbalance) > 0.5:
            side = "BUY" if signal.book_imbalance > 0 else "SELL"
            recommendations["book_pressure"] = f"Strong {side} pressure"

        return recommendations


# Global singleton
_microstructure: Optional[MicrostructureAnalyzer] = None


def get_microstructure_analyzer() -> MicrostructureAnalyzer:
    """Get or create global MicrostructureAnalyzer."""
    global _microstructure
    if _microstructure is None:
        _microstructure = MicrostructureAnalyzer()
    return _microstructure
