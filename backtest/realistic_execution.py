"""
Realistic execution handler for backtesting with market microstructure simulation.
Simulates realistic market conditions including slippage, market impact, and partial fills.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents an executed trade."""

    ticker: str
    quantity: float
    price: float
    timestamp: datetime
    execution_type: str = "MARKET"
    slippage: float = 0.0
    market_impact: float = 0.0
    transaction_cost: float = 0.0
    venue: str = "PRIMARY"
    partial: bool = False


class RealisticExecutionHandler:
    """
    Realistic execution handler that simulates market microstructure.

    Features:
    - Slippage modeling based on order size and market conditions
    - Market impact modeling
    - Liquidity constraints
    - ADV (Average Daily Volume) participation caps
    - Commission modeling
    """

    def __init__(
        self,
        slippage_bps: float = 2.0,
        impact_bps: float = 5.0,
        commission_bps: float = 1.0,
        max_adv_participation: float = 0.1,
        volatility_scalar: float = 1.0,
    ):
        """
        Initialize execution handler.

        Args:
            slippage_bps: Base slippage in basis points
            impact_bps: Base market impact in basis points per $1M
            commission_bps: Commission in basis points
            max_adv_participation: Maximum % of ADV we can execute
            volatility_scalar: Scalar for volatility-adjusted slippage
        """
        self.slippage_bps = slippage_bps
        self.impact_bps = impact_bps
        self.commission_bps = commission_bps
        self.max_adv_participation = max_adv_participation
        self.volatility_scalar = volatility_scalar

    def fill_order(
        self, order, bar, price_history: pd.Series, volume_history: pd.Series
    ) -> Optional[Trade]:
        """
        Simulate order execution with realistic market conditions.

        Args:
            order: Order object with ticker, quantity, order_type
            bar: BarData with OHLCV and timestamp
            price_history: Historical prices (last N bars)
            volume_history: Historical volumes (last N bars)

        Returns:
            Trade object if successfully filled, None otherwise
        """
        if order is None or bar is None:
            logger.warning("Invalid order or bar data")
            return None

        # Get ticker
        ticker = (
            order.ticker
            if hasattr(order, "ticker")
            else getattr(bar, "ticker", "UNKNOWN")
        )

        # Extract execution details
        quantity = order.quantity if hasattr(order, "quantity") else 0
        order_type = getattr(order, "order_type", "MARKET")

        # Handle string order types
        if isinstance(order_type, str):
            order_type_str = order_type
        else:
            order_type_str = (
                order_type.value if hasattr(order_type, "value") else str(order_type)
            )

        if quantity == 0:
            return None

        # Mid price (mid of bar)
        mid_price = (bar.open + bar.close) / 2

        # Calculate realized price based on order side
        is_buy = quantity > 0
        fill_price = self._calculate_fill_price(
            mid_price, abs(quantity), bar, price_history, volume_history, is_buy
        )

        # Calculate slippage and market impact
        slippage = self._calculate_slippage(
            mid_price, fill_price, price_history, is_buy
        )

        # Calculate market impact
        market_impact = self._calculate_market_impact(
            mid_price, abs(quantity), volume_history
        )

        # Calculate transaction cost
        transaction_cost = self._calculate_transaction_cost(fill_price, abs(quantity))

        # Check liquidity constraints
        can_fill, filled_qty = self._check_liquidity_constraint(
            abs(quantity), bar.volume, volume_history
        )

        if not can_fill:
            logger.debug(f"Liquidity constraint prevented full fill for {ticker}")
            filled_qty = filled_qty if filled_qty > 0 else abs(quantity) * 0.5

        # Create trade
        trade = Trade(
            ticker=ticker,
            quantity=filled_qty if is_buy else -filled_qty,
            price=fill_price,
            timestamp=bar.timestamp,
            execution_type=order_type_str,
            slippage=slippage,
            market_impact=market_impact,
            transaction_cost=transaction_cost,
            venue="PRIMARY",
            partial=filled_qty < abs(quantity),
        )

        logger.debug(
            f"Executed {ticker}: qty={trade.quantity} @ {fill_price:.2f}, "
            f"slippage={slippage:.4f} bps, impact={market_impact:.4f} bps"
        )

        return trade

    def _calculate_fill_price(
        self,
        mid_price: float,
        quantity: float,
        bar,
        price_history: pd.Series,
        volume_history: pd.Series,
        is_buy: bool,
    ) -> float:
        """Calculate the fill price considering market conditions."""

        # For buy orders, we pay the ask (higher)
        # For sell orders, we receive the bid (lower)
        bid_ask_spread = self._calculate_bid_ask_spread(mid_price, volume_history)

        if is_buy:
            base_price = mid_price + (bid_ask_spread / 2)
        else:
            base_price = mid_price - (bid_ask_spread / 2)

        # Add slippage based on size
        volatility = self._estimate_volatility(price_history)
        size_slippage = self._calculate_size_slippage(quantity, volume_history)

        total_slippage_pct = (self.slippage_bps + size_slippage) / 10000
        total_slippage_pct *= 1 + self.volatility_scalar * volatility

        if is_buy:
            fill_price = base_price * (1 + total_slippage_pct)
        else:
            fill_price = base_price * (1 - total_slippage_pct)

        return fill_price

    def _calculate_slippage(
        self,
        mid_price: float,
        fill_price: float,
        price_history: pd.Series,
        is_buy: bool,
    ) -> float:
        """Calculate slippage in basis points."""
        if mid_price == 0:
            return 0

        diff = abs(fill_price - mid_price) / mid_price
        return diff * 10000  # Convert to basis points

    def _calculate_market_impact(
        self, mid_price: float, quantity: float, volume_history: pd.Series
    ) -> float:
        """Calculate market impact in basis points based on ADV and notional."""

        avg_daily_volume = volume_history.mean() if len(volume_history) > 0 else 1000000
        notional = quantity * mid_price
        avg_notional = avg_daily_volume * mid_price

        if avg_notional == 0:
            return 0

        participation_pct = notional / avg_notional

        # Market impact increases with participation rate
        # Nonlinear: sqrt relationship
        impact_pct = self.impact_bps * np.sqrt(participation_pct) / 10000

        return impact_pct * 10000  # Return in basis points

    def _calculate_transaction_cost(self, fill_price: float, quantity: float) -> float:
        """Calculate transaction costs (commissions, fees, etc)."""
        notional = fill_price * quantity
        cost = notional * (self.commission_bps / 10000)
        return cost

    def _calculate_bid_ask_spread(
        self, mid_price: float, volume_history: pd.Series
    ) -> float:
        """Estimate bid-ask spread based on liquidity."""

        avg_volume = volume_history.mean() if len(volume_history) > 0 else 1000000

        # Spread tightens with higher volume
        base_spread_pct = 0.001  # 10 bps default

        if avg_volume < 100000:
            base_spread_pct = 0.005
        elif avg_volume > 10000000:
            base_spread_pct = 0.0001

        return mid_price * base_spread_pct

    def _calculate_size_slippage(
        self, quantity: float, volume_history: pd.Series
    ) -> float:
        """Calculate additional slippage based on order size."""

        avg_volume = volume_history.mean() if len(volume_history) > 0 else 1000000

        if avg_volume == 0:
            return 0

        participation = quantity / avg_volume

        # Nonlinear increase in slippage with participation
        size_slippage = max(0, min(50, 10 * (participation**1.5)))

        return size_slippage

    def _check_liquidity_constraint(
        self, quantity: float, bar_volume: float, volume_history: pd.Series
    ) -> Tuple[bool, float]:
        """Check if order can be filled under liquidity constraints."""

        avg_volume = volume_history.mean() if len(volume_history) > 0 else bar_volume
        max_participation = avg_volume * self.max_adv_participation

        can_fill = quantity <= max_participation
        filled_qty = min(quantity, max_participation)

        return can_fill, filled_qty

    def _estimate_volatility(self, price_history: pd.Series) -> float:
        """Estimate volatility from price history."""

        if len(price_history) < 2:
            return 0.02  # Default 2% volatility

        # Calculate returns
        returns = price_history.pct_change().dropna()

        if len(returns) == 0:
            return 0.02

        volatility = returns.std()

        return max(0.001, min(1.0, volatility))  # Bound between 0.1% and 100%
