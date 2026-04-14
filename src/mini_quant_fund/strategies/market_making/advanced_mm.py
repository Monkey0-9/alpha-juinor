"""
Advanced Market Making
======================

Sophisticated market making with:
- Inventory management
- Adverse selection protection
- Cross-asset market making
- Dynamic spread optimization
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketOrder:
    """Market order representation."""

    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    timestamp: float


class InventoryManagementModel:
    """
    Inventory risk management for market makers.

    Uses Avellaneda-Stoikov framework.
    """

    def __init__(
        self,
        risk_aversion: float = 0.1,
        inventory_target: float = 0,
        inventory_limit: float = 1000,
    ):
        self.risk_aversion = risk_aversion
        self.inventory_target = inventory_target
        self.inventory_limit = inventory_limit

    def compute_reservation_price(
        self,
        mid_price: float,
        current_inventory: float,
        volatility: float,
        time_to_close: float = 1.0,
    ) -> float:
        """
        Compute reservation price based on inventory.

        Args:
            mid_price: Current mid price
            current_inventory: Current inventory position
            volatility: Asset volatility
            time_to_close: Time to market close (in hours)

        Returns:
            Reservation price
        """
        # Avellaneda-Stoikov reservation price
        inventory_deviation = current_inventory - self.inventory_target

        price_adjustment = (
            self.risk_aversion * volatility**2 * time_to_close * inventory_deviation
        )

        reservation_price = mid_price - price_adjustment

        return reservation_price

    def compute_optimal_spread(
        self,
        mid_price: float,
        volatility: float,
        order_arrival_rate: float,
        time_to_close: float = 1.0,
    ) -> float:
        """
        Compute optimal bid-ask spread.

        Args:
            mid_price: Current mid price
            volatility: Asset volatility
            order_arrival_rate: Rate of market orders (per hour)
            time_to_close: Time to market close

        Returns:
            Optimal half-spread
        """
        # Avellaneda-Stoikov optimal spread
        gamma = self.risk_aversion

        optimal_half_spread = (
            gamma * volatility**2 * time_to_close
            + (2 / gamma) * np.log(1 + gamma / order_arrival_rate)
        )

        return optimal_half_spread / 2

    def get_quotes(
        self,
        mid_price: float,
        current_inventory: float,
        volatility: float,
        order_arrival_rate: float = 10,
        time_to_close: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Get bid and ask quotes.

        Args:
            mid_price: Current mid price
            current_inventory: Current inventory
            volatility: Volatility
            order_arrival_rate: Order arrival rate
            time_to_close: Time to close

        Returns:
            (bid_price, ask_price)
        """
        reservation_price = self.compute_reservation_price(
            mid_price, current_inventory, volatility, time_to_close
        )

        half_spread = self.compute_optimal_spread(
            mid_price, volatility, order_arrival_rate, time_to_close
        )

        bid = reservation_price - half_spread
        ask = reservation_price + half_spread

        return (bid, ask)


class AdverseSelectionProtection:
    """
    Protects against adverse selection from informed traders.

    Uses:
    - Order flow toxicity detection
    - Dynamic quote adjustment
    - Trade size limits
    """

    def __init__ (self, toxicity_threshold: float = 0.6):
        self.toxicity_threshold = toxicity_threshold
        self.recent_trades: List[MarketOrder] = []

    def estimate_order_flow_toxicity(
        self,
        recent_price_moves: List[float],
        recent_trade_sizes: List[float],
    ) -> float:
        """
        Estimate toxicity of order flow (VPIN-inspired).

        Args:
            recent_price_moves: Recent price movements
            recent_trade_sizes: Recent trade sizes

        Returns:
            Toxicity score [0, 1], higher = more toxic
        """
        if len(recent_price_moves) < 2:
            return 0.5

        # Volume-synchronized probability of informed trading
        price_changes = np.diff(recent_price_moves)

        # Classify as buy or sell based on price direction
        buy_volume = sum(
            size
            for size, change in zip(recent_trade_sizes, price_changes)
            if change > 0
        )
        sell_volume = sum(
            size
            for size, change in zip(recent_trade_sizes, price_changes)
            if change < 0
        )

        total_volume = buy_volume + sell_volume

        if total_volume == 0:
            return 0.5

        # Order imbalance
        imbalance = abs(buy_volume - sell_volume) / total_volume

        # Simple toxicity metric
        toxicity = np.clip(imbalance, 0, 1)

        return toxicity

    def adjust_quote_for_toxicity(
        self,
        base_bid: float,
        base_ask: float,
        toxicity: float,
    ) -> Tuple[float, float]:
        """
        Widen spread based on toxicity.

        Args:
            base_bid: Base bid price
            base_ask: Base ask price
            toxicity: Toxicity score

        Returns:
            (adjusted_bid, adjusted_ask)
        """
        if toxicity > self.toxicity_threshold:
            # Widen spread
            mid = (base_bid + base_ask) / 2
            spread = base_ask - base_bid

            # Increase spread by toxicity factor
            widening_factor = 1 + 2 * (toxicity - self.toxicity_threshold)
            new_spread = spread * widening_factor

            adjusted_bid = mid - new_spread / 2
            adjusted_ask = mid + new_spread / 2

            return (adjusted_bid, adjusted_ask)

        return (base_bid, base_ask)


class CrossAssetMarketMaker:
    """
    Market making across correlated assets.

    Exploits relative value relationships.
    """

    def __init__(self):
        self.correlations: Dict[Tuple[str, str], float] = {}

    def set_correlation(self, asset1: str, asset2: str, correlation: float):
        """Set correlation between two assets."""
        key = tuple(sorted([asset1, asset2]))
        self.correlations[key] = correlation

    def get_hedge_ratio(self, asset1: str, asset2: str) -> float:
        """
        Get hedge ratio for cross-asset market making.

        Args:
            asset1: First asset
            asset2: Second asset

        Returns:
            Hedge ratio (units of asset2 per unit of asset1)
        """
        key = tuple(sorted([asset1, asset2]))
        correlation = self.correlations.get(key, 0)

        # Simplified: assume beta = correlation
        # In practice, would use regression
        hedge_ratio = correlation

        return hedge_ratio

    def compute_cross_asset_quotes(
        self,
        primary_asset: str,
        hedging_asset: str,
        primary_mid: float,
        hedging_mid: float,
        primary_spread: float,
    ) -> Tuple[float, float, float]:
        """
        Compute quotes accounting for hedging costs.

        Args:
            primary_asset: Asset to quote
            hedging_asset: Asset used for hedging
            primary_mid: Mid price of primary
            hedging_mid: Mid price of hedging asset
            primary_spread: Base spread for primary

        Returns:
            (bid, ask, hedge_quantity)
        """
        hedge_ratio = self.get_hedge_ratio(primary_asset, hedging_asset)

        # Hedging cost
        hedging_spread_cost = 0.0001 * hedging_mid * abs(hedge_ratio)

        # Adjust spread to account for hedging cost
        adjusted_spread = primary_spread + hedging_spread_cost

        bid = primary_mid - adjusted_spread / 2
        ask = primary_mid + adjusted_spread / 2

        return (bid, ask, hedge_ratio)


class AdvancedMarketMakingStrategy:
    """
    Complete market making strategy combining all components.
    """

    def __init__(
        self,
        symbol: str,
        risk_aversion: float = 0.1,
        inventory_limit: float = 1000,
    ):
        self.symbol = symbol
        self.inventory_manager = InventoryManagementModel(
            risk_aversion=risk_aversion,
            inventory_limit=inventory_limit,
        )
        self.adverse_selection = AdverseSelectionProtection()
        self.cross_asset_mm = CrossAssetMarketMaker()

        self.current_inventory = 0
        self.trades_history: List[MarketOrder] = []

    def get_quotes(
        self,
        mid_price: float,
        volatility: float,
        recent_price_moves: List[float],
        recent_trade_sizes: List[float],
    ) -> Dict[str, float]:
        """
        Get market making quotes.

        Args:
            mid_price: Current mid price
            volatility: Volatility
            recent_price_moves: Recent price movements
            recent_trade_sizes: Recent trade sizes

        Returns:
            Dictionary with bid, ask, spread, inventory info
        """
        # Step 1: Get base quotes from inventory model
        base_bid, base_ask = self.inventory_manager.get_quotes(
            mid_price=mid_price,
            current_inventory=self.current_inventory,
            volatility=volatility,
        )

        # Step 2: Adjust for adverse selection
        toxicity = self.adverse_selection.estimate_order_flow_toxicity(
            recent_price_moves, recent_trade_sizes
        )

        adjusted_bid, adjusted_ask = self.adverse_selection.adjust_quote_for_toxicity(
            base_bid, base_ask, toxicity
        )

        return {
            "bid": adjusted_bid,
            "ask": adjusted_ask,
            "mid": mid_price,
            "spread": adjusted_ask - adjusted_bid,
            "spread_bps": ((adjusted_ask - adjusted_bid) / mid_price) * 10000,
            "inventory": self.current_inventory,
            "toxicity": toxicity,
        }

    def execute_trade(self, order: MarketOrder):
        """
        Execute trade and update inventory.

        Args:
            order: Market order to execute
        """
        if order.side == "BUY":
            self.current_inventory += order.quantity
        else:
            self.current_inventory -= order.quantity

        self.trades_history.append(order)

        logger.info(
            f"Executed {order.side} {order.quantity} {self.symbol}, inventory now: {self.current_inventory}"
        )
