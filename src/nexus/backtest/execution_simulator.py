import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ExecutionType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    DMA = "dma"  # Direct Market Access

class LiquidityProfile(Enum):
    HIGH = "high_liquidity"
    MEDIUM = "medium_liquidity"
    LOW = "low_liquidity"
    ILLIQUID = "illiquid"

@dataclass
class ExecutionOrder:
    """Represents a trading order with execution parameters."""
    ticker: str
    quantity: float
    side: str  # 'buy' or 'sell'
    order_type: ExecutionType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ExecutionResult:
    """Result of an order execution."""
    order: ExecutionOrder
    executed_quantity: float
    executed_price: float
    slippage: float
    market_impact: float
    transaction_costs: float
    execution_time: datetime
    venue: str
    success: bool
    rejection_reason: Optional[str] = None

class InstitutionalExecutionSimulator:
    """
    INSTITUTIONAL-GRADE EXECUTION SIMULATOR
    Models realistic market impact, slippage, and execution dynamics.
    Supports DMA (Direct Market Access) simulation for top-tier realism.
    """

    def __init__(self, market_data_router=None, risk_manager=None):
        self.market_data = market_data_router
        self.risk_manager = risk_manager

        # Execution Parameters (Institutional-grade defaults)
        self.base_spread_bps = 0.5  # 0.5 bps base spread
        self.market_impact_model = "square_root"  # Almgren-Chriss model
        self.max_participation_rate = 0.10  # Max 10% of volume
        self.dma_enabled = True

        # Venue-specific parameters
        self.venue_configs = {
            'NYSE': {'liquidity': LiquidityProfile.HIGH, 'commission_bps': 0.3},
            'NASDAQ': {'liquidity': LiquidityProfile.HIGH, 'commission_bps': 0.3},
            'DMA': {'liquidity': LiquidityProfile.HIGH, 'commission_bps': 0.1},  # Lower costs for DMA
            'DARK_POOL': {'liquidity': LiquidityProfile.MEDIUM, 'commission_bps': 0.05},
            'OTC': {'liquidity': LiquidityProfile.LOW, 'commission_bps': 1.0}
        }

        # Market microstructure parameters
        self.order_book_depth = 10  # Levels to model
        self.min_order_size = 100  # Minimum order size
        self.max_order_size_pct = 0.05  # Max 5% of average daily volume

        logger.info("Institutional Execution Simulator initialized")

    def execute_order(self, order: ExecutionOrder, market_data: pd.DataFrame) -> ExecutionResult:
        """
        Execute a single order with realistic market impact and slippage modeling.
        """
        try:
            # 1. Pre-execution validation
            if not self._validate_order(order, market_data):
                return ExecutionResult(
                    order=order,
                    executed_quantity=0,
                    executed_price=0,
                    slippage=0,
                    market_impact=0,
                    transaction_costs=0,
                    execution_time=datetime.now(),
                    venue="REJECTED",
                    success=False,
                    rejection_reason="Order validation failed"
                )

            # 2. Determine execution venue
            venue = self._select_execution_venue(order, market_data)

            # 3. Calculate execution price with slippage and market impact
            execution_price, slippage, market_impact = self._calculate_execution_price(
                order, market_data, venue
            )

            # 4. Calculate transaction costs
            transaction_costs = self._calculate_transaction_costs(order, venue)

            # 5. Apply execution timing (simulated latency)
            execution_time = order.timestamp + timedelta(milliseconds=np.random.randint(10, 500))

            # 6. Handle partial fills for large orders
            executed_quantity = self._calculate_executed_quantity(order, market_data, venue)

            return ExecutionResult(
                order=order,
                executed_quantity=executed_quantity,
                executed_price=execution_price,
                slippage=slippage,
                market_impact=market_impact,
                transaction_costs=transaction_costs,
                execution_time=execution_time,
                venue=venue,
                success=True
            )

        except Exception as e:
            logger.error(f"Order execution failed for {order.ticker}: {e}")
            return ExecutionResult(
                order=order,
                executed_quantity=0,
                executed_price=0,
                slippage=0,
                market_impact=0,
                transaction_costs=0,
                execution_time=datetime.now(),
                venue="ERROR",
                success=False,
                rejection_reason=str(e)
            )

    def execute_portfolio_orders(self, orders: List[ExecutionOrder], market_data: Dict[str, pd.DataFrame]) -> List[ExecutionResult]:
        """
        Execute multiple orders with portfolio-level considerations.
        Accounts for cross-impact and optimal execution sequencing.
        """
        results = []

        # Sort orders by urgency and size (large orders first for better execution)
        sorted_orders = sorted(orders, key=lambda x: abs(x.quantity), reverse=True)

        for order in sorted_orders:
            if order.ticker in market_data:
                result = self.execute_order(order, market_data[order.ticker])
                results.append(result)

                # Update market data to reflect executed order's market impact
                if result.success and result.market_impact > 0:
                    self._apply_market_impact_to_data(market_data[order.ticker], result)

        return results

    def _validate_order(self, order: ExecutionOrder, market_data: pd.DataFrame) -> bool:
        """Validate order parameters and market conditions."""
        # Check minimum order size
        if abs(order.quantity) < self.min_order_size:
            logger.warning(f"Order size {order.quantity} below minimum {self.min_order_size}")
            return False

        # Check market data availability
        if market_data.empty:
            logger.warning(f"No market data available for {order.ticker}")
            return False

        # Check liquidity constraints
        avg_volume = market_data['Volume'].rolling(20).mean().iloc[-1]
        if avg_volume > 0:
            order_size_pct = abs(order.quantity) / avg_volume
            if order_size_pct > self.max_order_size_pct:
                logger.warning(f"Order size {order_size_pct:.2%} exceeds max {self.max_order_size_pct:.2%} of ADV")
                return False

        # Validate limit/stop prices
        if order.order_type == ExecutionType.LIMIT and order.limit_price:
            current_price = market_data['Close'].iloc[-1]
            if order.side == 'buy' and order.limit_price < current_price * 0.95:  # Too far below market
                return False
            if order.side == 'sell' and order.limit_price > current_price * 1.05:  # Too far above market
                return False

        return True

    def _select_execution_venue(self, order: ExecutionOrder, market_data: pd.DataFrame) -> str:
        """Select optimal execution venue based on order characteristics."""
        # DMA for institutional orders
        if self.dma_enabled and abs(order.quantity) > 10000:
            return "DMA"

        # Dark pools for large orders to minimize market impact
        if abs(order.quantity) > 50000:
            return "DARK_POOL"

        # Determine primary exchange
        ticker = order.ticker
        if ticker.endswith('.O') or any(x in ticker for x in ['AAPL', 'MSFT', 'GOOGL']):
            return "NASDAQ"
        else:
            return "NYSE"

    def _calculate_execution_price(self, order: ExecutionOrder, market_data: pd.DataFrame, venue: str) -> Tuple[float, float, float]:
        """
        Calculate execution price with realistic slippage and market impact.
        Uses Almgren-Chriss market impact model.
        """
        current_price = market_data['Close'].iloc[-1]
        spread = self._calculate_spread(market_data, venue)

        # Base execution price
        if order.order_type == ExecutionType.MARKET:
            # Market orders execute at mid + half spread + slippage
            base_price = current_price + (spread / 2) * (1 if order.side == 'buy' else -1)
        elif order.order_type == ExecutionType.LIMIT and order.limit_price:
            base_price = order.limit_price
        else:
            base_price = current_price

        # Calculate market impact (Almgren-Chriss model)
        market_impact = self._calculate_market_impact(order, market_data, venue)

        # Add slippage (temporary price movement)
        slippage = self._calculate_slippage(order, market_data, venue)

        # Final execution price
        execution_price = base_price + market_impact + slippage

        # Ensure price doesn't go negative or unrealistic
        execution_price = max(execution_price, current_price * 0.5)

        return execution_price, slippage, market_impact

    def _calculate_market_impact(self, order: ExecutionOrder, market_data: pd.DataFrame, venue: str) -> float:
        """Calculate permanent market impact using Almgren-Chriss model."""
        try:
            # Get average daily volume
            adv = market_data['Volume'].rolling(20).mean().iloc[-1]
            if adv <= 0:
                return 0

            # Order size as percentage of ADV
            participation_rate = abs(order.quantity) / adv
            participation_rate = min(participation_rate, self.max_participation_rate)

            # Volatility-based impact
            returns = market_data['Close'].pct_change().rolling(20).std().iloc[-1]
            volatility = max(returns, 0.01)  # Minimum 1% volatility

            # Almgren-Chriss impact formula: impact ∝ σ * (Q/ADV)^(1/2)
            if self.market_impact_model == "square_root":
                impact_magnitude = volatility * np.sqrt(participation_rate)
            else:
                # Linear model for comparison
                impact_magnitude = volatility * participation_rate

            # Adjust for venue liquidity
            venue_config = self.venue_configs.get(venue, self.venue_configs['NYSE'])
            liquidity_multiplier = {
                LiquidityProfile.HIGH: 0.8,
                LiquidityProfile.MEDIUM: 1.0,
                LiquidityProfile.LOW: 1.5,
                LiquidityProfile.ILLIQUID: 2.0
            }.get(venue_config['liquidity'], 1.0)

            impact = impact_magnitude * liquidity_multiplier * (1 if order.side == 'buy' else -1)

            return impact

        except Exception as e:
            logger.warning(f"Market impact calculation failed: {e}")
            return 0

    def _calculate_slippage(self, order: ExecutionOrder, market_data: pd.DataFrame, venue: str) -> float:
        """Calculate temporary slippage due to order book dynamics."""
        try:
            # Base slippage from bid-ask spread
            spread = self._calculate_spread(market_data, venue)

            # Order book slippage (deeper orders have more slippage)
            order_book_slippage = spread * np.random.uniform(0.1, 0.5)

            # Time-based slippage (market orders hit multiple levels)
            if order.order_type == ExecutionType.MARKET:
                time_slippage = spread * np.random.uniform(0.2, 1.0)
            else:
                time_slippage = 0

            # Size-based slippage (larger orders walk the book)
            size_multiplier = min(abs(order.quantity) / 10000, 3.0)  # Cap at 3x
            size_slippage = spread * size_multiplier * 0.5

            total_slippage = order_book_slippage + time_slippage + size_slippage

            return total_slippage * (1 if order.side == 'buy' else -1)

        except Exception as e:
            logger.warning(f"Slippage calculation failed: {e}")
            return 0

    def _calculate_spread(self, market_data: pd.DataFrame, venue: str) -> float:
        """Calculate effective bid-ask spread."""
        try:
            # Use high-low range as proxy for spread
            recent_data = market_data.tail(20)
            avg_spread = (recent_data['High'] - recent_data['Low']).mean() / recent_data['Close'].mean()

            # Venue-specific spread adjustments
            venue_config = self.venue_configs.get(venue, self.venue_configs['NYSE'])
            if venue_config['liquidity'] == LiquidityProfile.HIGH:
                spread_multiplier = 0.8
            elif venue_config['liquidity'] == LiquidityProfile.MEDIUM:
                spread_multiplier = 1.2
            else:
                spread_multiplier = 2.0

            effective_spread = max(avg_spread * spread_multiplier, self.base_spread_bps / 10000)

            return effective_spread

        except Exception:
            return self.base_spread_bps / 10000  # Fallback to base spread

    def _calculate_transaction_costs(self, order: ExecutionOrder, venue: str) -> float:
        """Calculate total transaction costs including commissions and fees."""
        venue_config = self.venue_configs.get(venue, self.venue_configs['NYSE'])

        # Commission (bps of notional value)
        notional_value = abs(order.quantity) * 100  # Assume $100 per share for simplicity
        commission = notional_value * (venue_config['commission_bps'] / 10000)

        # Exchange fees (simplified)
        exchange_fee = notional_value * 0.0001  # 0.01 bps

        # SEC fees (for US equities)
        sec_fee = notional_value * 0.000005  # 0.0005 bps

        return commission + exchange_fee + sec_fee

    def _calculate_executed_quantity(self, order: ExecutionOrder, market_data: pd.DataFrame, venue: str) -> float:
        """Calculate actual executed quantity (may be partial for large orders)."""
        # For simplicity, assume full execution in backtest
        # In live trading, this would handle partial fills
        return order.quantity

    def _apply_market_impact_to_data(self, market_data: pd.DataFrame, result: ExecutionResult):
        """Apply the market impact of executed order to subsequent market data."""
        # This would modify the market data to reflect price impact
        # For backtesting realism, we could adjust subsequent prices
        pass

    def get_execution_metrics(self, results: List[ExecutionResult]) -> Dict[str, float]:
        """Calculate execution quality metrics."""
        if not results:
            return {}

        successful_results = [r for r in results if r.success]

        metrics = {
            'total_orders': len(results),
            'successful_orders': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_slippage_bps': 0,
            'avg_market_impact_bps': 0,
            'avg_transaction_costs_bps': 0,
            'total_executed_value': 0
        }

        if successful_results:
            executed_values = [abs(r.executed_quantity) * r.executed_price for r in successful_results]
            metrics['total_executed_value'] = sum(executed_values)

            # Calculate average costs in basis points
            avg_price = sum(r.executed_price for r in successful_results) / len(successful_results)
            if avg_price > 0:
                metrics['avg_slippage_bps'] = (sum(r.slippage for r in successful_results) / len(successful_results)) / avg_price * 10000
                metrics['avg_market_impact_bps'] = (sum(r.market_impact for r in successful_results) / len(successful_results)) / avg_price * 10000
                metrics['avg_transaction_costs_bps'] = (sum(r.transaction_costs for r in successful_results) / len(successful_results)) / avg_price * 10000

        return metrics
