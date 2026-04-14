"""
Regulatory Automation
====================

MiFID II compliance, best execution analytics, and trade surveillance.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Trade record for compliance."""

    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    venue: str
    client_id: str


class MiFIDIICompliance:
    """
    MiFID II compliance automation.

    Requirements:
    - Best execution reporting
    - Transaction reporting
    - Clock synchronization
    - Record keeping
    """

    def __init__(self):
        self.trades: List[TradeRecord] = []

    def add_trade(self, trade: TradeRecord):
        """Add trade for compliance tracking."""
        self.trades.append(trade)

    def generate_transaction_report(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate MiFID II transaction report.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with required fields
        """
        filtered_trades = [
            t for t in self.trades if start_date <= t.timestamp <= end_date
        ]

        if not filtered_trades:
            return pd.DataFrame()

        data = {
            "Trading Day": [t.timestamp.date() for t in filtered_trades],
            "Trading Time": [t.timestamp.time() for t in filtered_trades],
            "Instrument": [t.symbol for t in filtered_trades],
            "Buy/Sell": [t.side for t in filtered_trades],
            "Quantity": [t.quantity for t in filtered_trades],
            "Price": [t.price for t in filtered_trades],
            "Venue": [t.venue for t in filtered_trades],
            "Transaction ID": [t.trade_id for t in filtered_trades],
            "Client ID": [t.client_id for t in filtered_trades],
        }

        return pd.DataFrame(data)


class BestExecutionAnalytics:
    """
    Best execution analytics for MiFID II.

    Analyzes:
    - Price improvement
    - Effective spread
    - Venue quality
    """

    def __init__(self):
        self.trades: List[TradeRecord] = []
        self.benchmarks: Dict[str, float] = {}  # Mid prices at trade time

    def add_trade(self, trade: TradeRecord, benchmark_mid: float):
        """
        Add trade with benchmark price.

        Args:
            trade: Trade record
            benchmark_mid: Mid price at trade time
        """
        self.trades.append(trade)
        self.benchmarks[trade.trade_id] = benchmark_mid

    def calculate_price_improvement(self, trade: TradeRecord) -> float:
        """
        Calculate price improvement in basis points.

        Args:
            trade: Trade record

        Returns:
            Price improvement (positive = better than benchmark)
        """
        benchmark = self.benchmarks.get(trade.trade_id)
        if benchmark is None:
            return 0.0

        if trade.side == "BUY":
            improvement = benchmark - trade.price
        else:  # SELL
            improvement = trade.price - benchmark

        # Convert to basis points
        improvement_bps = (improvement / benchmark) * 10000

        return improvement_bps

    def generate_best_execution_report(self) -> Dict[str, any]:
        """
        Generate best execution report.

        Returns:
            Dictionary with execution quality metrics
        """
        if not self.trades:
            return {}

        # Calculate price improvement for all trades
        improvements = [self.calculate_price_improvement(t) for t in self.trades]

        # Group by venue
        venue_stats = {}
        for trade, improvement in zip(self.trades, improvements):
            if trade.venue not in venue_stats:
                venue_stats[trade.venue] = {"improvements": [], "volumes": []}

            venue_stats[trade.venue]["improvements"].append(improvement)
            venue_stats[trade.venue]["volumes"].append(
                trade.quantity * trade.price
            )

        # Compute venue-level statistics
        venue_quality = {}
        for venue, stats in venue_stats.items():
            venue_quality[venue] = {
                "avg_price_improvement_bps": np.mean(stats["improvements"]),
                "total_volume": sum(stats["volumes"]),
                "trade_count": len(stats["improvements"]),
            }

        return {
            "overall_avg_improvement_bps": np.mean(improvements),
            "total_trades": len(self.trades),
            "venue_quality": venue_quality,
        }


class TradeSurveillance:
    """
    Automated trade surveillance for market abuse detection.

    Patterns:
    - Wash trading
    - Layering/spoofing
    - Ramping
    - Front-running
    """

    def __init__(self):
        self.alerts: List[Dict] = []

    def detect_wash_trading(
        self, trades: List[TradeRecord], time_window_seconds: int = 60
    ) -> List[Dict]:
        """
        Detect potential wash trading.

        Wash trading = buying and selling same instrument rapidly.

        Args:
            trades: List of trades
            time_window_seconds: Time window for detection

        Returns:
            List of alerts
        """
        alerts = []

        # Group by symbol and client
        by_symbol_client = {}
        for trade in trades:
            key = (trade.symbol, trade.client_id)
            if key not in by_symbol_client:
                by_symbol_client[key] = []
            by_symbol_client[key].append(trade)

        # Check for matching buy/sell pairs
        for (symbol, client), client_trades in by_symbol_client.items():
            client_trades.sort(key=lambda t: t.timestamp)

            for i, trade1 in enumerate(client_trades):
                for trade2 in client_trades[i + 1 :]:
                    # Check if within time window
                    time_diff = (trade2.timestamp - trade1.timestamp).total_seconds()
                    if time_diff > time_window_seconds:
                        break

                    # Check if opposite sides with similar quantities
                    if (
                        trade1.side != trade2.side
                        and abs(trade1.quantity - trade2.quantity) < 0.1 * trade1.quantity
                    ):
                        alerts.append(
                            {
                                "type": "WASH_TRADING",
                                "symbol": symbol,
                                "client": client,
                                "trade_pair": [trade1.trade_id, trade2.trade_id],
                                "severity": "HIGH",
                                "description": f"Matched buy/sell within {time_diff:.0f}s",
                            }
                        )

        self.alerts.extend(alerts)
        return alerts

    def detect_layering(
        self, orders: List[Dict], execution_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Detect layering/spoofing.

        Layering = placing large orders to create false impression,
        then executing small order on other side.

        Args:
            orders: List of order dictionaries
            execution_threshold: % of orders that execute

        Returns:
            List of alerts
        """
        alerts = []

        # Group by symbol
        by_symbol = {}
        for order in orders:
            symbol = order.get("symbol")
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(order)

        for symbol, symbol_orders in by_symbol.items():
            # Count cancelled vs executed
            cancelled = sum(1 for o in symbol_orders if o.get("status") == "CANCELLED")
            executed = sum(1 for o in symbol_orders if o.get("status") == "FILLED")

            total = len(symbol_orders)
            if total > 10 and (executed / total) < execution_threshold:
                alerts.append(
                    {
                        "type": "LAYERING",
                        "symbol": symbol,
                        "severity": "MEDIUM",
                        "description": f"{cancelled}/{total} orders cancelled",
                        "execution_rate": executed / total,
                    }
                )

        self.alerts.extend(alerts)
        return alerts

    def get_alerts(self, severity: Optional[str] = None) -> List[Dict]:
        """Get surveillance alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self.alerts if a.get("severity") == severity]
        return self.alerts
