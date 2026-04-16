"""
Multi-Asset Trading System - Futures, Options, Forex, and Equities
Production-grade implementation supporting all major asset classes
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Supported asset classes"""
    EQUITY = "EQUITY"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"
    BONDS = "BONDS"
    COMMODITIES = "COMMODITIES"

class OrderType(Enum):
    """Order types for different asset classes"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    POV = "POV"  # Percentage of Volume
    MOC = "MOC"  # Market on Close
    MOO = "MOO"  # Market on Open

class TimeInForce(Enum):
    """Time in force options"""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date

@dataclass
class ContractSpecification:
    """Contract specifications for different asset classes"""
    symbol: str
    asset_class: AssetClass
    exchange: str
    currency: str
    multiplier: float = 1.0
    min_tick: float = 0.01
    contract_size: int = 1
    expiration_date: Optional[datetime] = None
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # CALL, PUT
    underlying: Optional[str] = None
    trading_hours: Optional[Dict[str, str]] = None
    margin_requirement: float = 0.0
    commission_per_contract: float = 0.0

@dataclass
class MultiAssetOrder:
    """Multi-asset order specification"""
    order_id: str
    contract: ContractSpecification
    side: str  # BUY, SELL
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    exchange: Optional[str] = None
    algo_params: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate order after creation"""
        self._validate_order()

    def _validate_order(self):
        """Validate order specifications"""
        if self.contract.asset_class == AssetClass.OPTIONS:
            if not self.contract.strike_price or not self.contract.option_type:
                raise ValueError("Options require strike_price and option_type")
        elif self.contract.asset_class == AssetClass.FUTURES:
            if not self.contract.expiration_date:
                raise ValueError("Futures require expiration_date")

        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require price")

        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require stop_price")

class MultiAssetExecutionEngine:
    """Production multi-asset execution engine"""

    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, MultiAssetOrder] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.contract_specs: Dict[str, ContractSpecification] = {}
        self._initialize_contract_specs()

    def _initialize_contract_specs(self):
        """Initialize contract specifications for major assets"""

        # Equities
        self.contract_specs["AAPL"] = ContractSpecification(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            exchange="NASDAQ",
            currency="USD",
            min_tick=0.01,
            commission_per_contract=0.005
        )

        # Futures (ES - S&P 500 E-mini)
        self.contract_specs["ES"] = ContractSpecification(
            symbol="ES",
            asset_class=AssetClass.FUTURES,
            exchange="CME",
            currency="USD",
            multiplier=50.0,
            min_tick=0.25,
            contract_size=1,
            expiration_date=datetime(2024, 12, 31),
            margin_requirement=12000.0,
            commission_per_contract=2.50
        )

        # Options (AAPL options)
        self.contract_specs["AAPL_CALL_150"] = ContractSpecification(
            symbol="AAPL_CALL_150",
            asset_class=AssetClass.OPTIONS,
            exchange="CBOE",
            currency="USD",
            multiplier=100.0,
            min_tick=0.01,
            contract_size=100,
            strike_price=150.0,
            option_type="CALL",
            underlying="AAPL",
            expiration_date=datetime(2024, 12, 20),
            commission_per_contract=0.65
        )

        # Forex (EUR/USD)
        self.contract_specs["EURUSD"] = ContractSpecification(
            symbol="EURUSD",
            asset_class=AssetClass.FOREX,
            exchange="FOREX",
            currency="USD",
            multiplier=100000.0,  # Standard lot
            min_tick=0.00001,
            commission_per_contract=0.00002  # 2 pips
        )

        # Crypto (BTC/USD)
        self.contract_specs["BTCUSD"] = ContractSpecification(
            symbol="BTCUSD",
            asset_class=AssetClass.CRYPTO,
            exchange="COINBASE",
            currency="USD",
            multiplier=1.0,
            min_tick=0.01,
            commission_per_contract=0.001  # 0.1%
        )

    def place_order(self, order: MultiAssetOrder) -> Dict[str, Any]:
        """Place a multi-asset order"""
        try:
            # Validate order
            self._validate_margin(order)

            # Store order
            self.orders[order.order_id] = order

            # Simulate execution
            execution_result = self._execute_order(order)

            # Update positions and capital
            self._update_position(order, execution_result)

            # Record execution
            self.execution_history.append(execution_result)

            return execution_result

        except Exception as e:
            logger.error(f"Error placing order {order.order_id}: {e}")
            return {
                "order_id": order.order_id,
                "status": "REJECTED",
                "error": str(e),
                "timestamp": datetime.now()
            }

    def _validate_margin(self, order: MultiAssetOrder):
        """Validate margin requirements"""
        contract = order.contract
        required_margin = 0.0

        if contract.asset_class == AssetClass.FUTURES:
            required_margin = contract.margin_requirement * abs(order.quantity)
        elif contract.asset_class == AssetClass.OPTIONS:
            # Simplified options margin calculation
            required_margin = contract.strike_price * contract.multiplier * abs(order.quantity) * 0.2

        if required_margin > self.current_capital:
            raise ValueError(f"Insufficient margin: Required ${required_margin:,.2f}, Available ${self.current_capital:,.2f}")

    def _execute_order(self, order: MultiAssetOrder) -> Dict[str, Any]:
        """Execute order with asset-specific logic"""
        contract = order.contract

        # Get market price (simulated)
        market_price = self._get_market_price(order)

        # Calculate execution price
        execution_price = self._calculate_execution_price(order, market_price)

        # Calculate commissions
        commission = self._calculate_commission(order, execution_price)

        # Calculate total cost
        total_value = execution_price * order.quantity * contract.multiplier
        total_cost = total_value + commission

        # Check capital availability
        if order.side == "BUY" and total_cost > self.current_capital:
            raise ValueError(f"Insufficient capital: Required ${total_cost:,.2f}, Available ${self.current_capital:,.2f}")

        return {
            "order_id": order.order_id,
            "status": "FILLED",
            "symbol": contract.symbol,
            "asset_class": contract.asset_class.value,
            "side": order.side,
            "quantity": order.quantity,
            "execution_price": execution_price,
            "market_price": market_price,
            "total_value": total_value,
            "commission": commission,
            "total_cost": total_cost,
            "timestamp": datetime.now(),
            "exchange": contract.exchange
        }

    def _get_market_price(self, order: MultiAssetOrder) -> float:
        """Get market price for the asset"""
        contract = order.contract

        # Simulated market prices
        base_prices = {
            AssetClass.EQUITY: 150.0,
            AssetClass.FUTURES: 4500.0,
            AssetClass.OPTIONS: 5.0,
            AssetClass.FOREX: 1.0850,
            AssetClass.CRYPTO: 45000.0
        }

        base_price = base_prices.get(contract.asset_class, 100.0)

        # Add some randomness
        price = base_price * (1 + np.random.normal(0, 0.01))

        # Round to minimum tick
        price = round(price / contract.min_tick) * contract.min_tick

        return price

    def _get_market_price_for_symbol(self, symbol: str) -> float:
        """Get market price for a symbol"""
        contract = self.contract_specs.get(symbol)
        if not contract:
            return 100.0

        # Simulated market prices
        base_prices = {
            AssetClass.EQUITY: 150.0,
            AssetClass.FUTURES: 4500.0,
            AssetClass.OPTIONS: 5.0,
            AssetClass.FOREX: 1.0850,
            AssetClass.CRYPTO: 45000.0
        }

        base_price = base_prices.get(contract.asset_class, 100.0)

        # Add some randomness
        price = base_price * (1 + np.random.normal(0, 0.01))

        # Round to minimum tick
        price = round(price / contract.min_tick) * contract.min_tick

        return price

    def _calculate_execution_price(self, order: MultiAssetOrder, market_price: float) -> float:
        """Calculate execution price based on order type"""
        if order.order_type == OrderType.MARKET:
            # Market orders get current price with slippage
            slippage = np.random.normal(0, 0.001)  # 0.1% slippage
            if order.side == "BUY":
                return market_price * (1 + abs(slippage))
            else:
                return market_price * (1 - abs(slippage))

        elif order.order_type == OrderType.LIMIT:
            # Limit orders get limit price or better
            if order.side == "BUY" and order.price >= market_price:
                return market_price  # Get better price
            elif order.side == "SELL" and order.price <= market_price:
                return market_price  # Get better price
            else:
                return order.price  # Use limit price

        return market_price

    def _calculate_commission(self, order: MultiAssetOrder, execution_price: float) -> float:
        """Calculate commission based on asset class"""
        contract = order.contract

        if contract.asset_class == AssetClass.EQUITY:
            return contract.commission_per_contract * abs(order.quantity)
        elif contract.asset_class == AssetClass.FUTURES:
            return contract.commission_per_contract * abs(order.quantity)
        elif contract.asset_class == AssetClass.OPTIONS:
            return contract.commission_per_contract * abs(order.quantity)
        elif contract.asset_class == AssetClass.FOREX:
            return execution_price * order.quantity * contract.commission_per_contract
        elif contract.asset_class == AssetClass.CRYPTO:
            return execution_price * order.quantity * contract.commission_per_contract
        else:
            return 0.0

    def _update_position(self, order: MultiAssetOrder, execution_result: Dict[str, Any]):
        """Update positions after execution"""
        symbol = execution_result["symbol"]
        side = execution_result["side"]
        quantity = execution_result["quantity"]
        execution_price = execution_result["execution_price"]
        total_cost = execution_result["total_cost"]

        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0.0,
                "avg_price": 0.0,
                "total_cost": 0.0,
                "asset_class": execution_result["asset_class"],
                "exchange": execution_result["exchange"]
            }

        position = self.positions[symbol]

        if side == "BUY":
            # Update position
            old_quantity = position["quantity"]
            old_cost = position["total_cost"]

            new_quantity = old_quantity + quantity
            new_cost = old_cost + total_cost

            position["quantity"] = new_quantity
            position["total_cost"] = new_cost
            position["avg_price"] = new_cost / new_quantity if new_quantity > 0 else 0

            # Update capital
            self.current_capital -= total_cost

        else:  # SELL
            # Check if we have enough position
            if position["quantity"] < quantity:
                raise ValueError(f"Insufficient position: Have {position['quantity']}, trying to sell {quantity}")

            # Update position
            old_quantity = position["quantity"]
            old_cost = position["total_cost"]

            new_quantity = old_quantity - quantity
            cost_basis = (old_cost / old_quantity) * quantity if old_quantity > 0 else 0

            position["quantity"] = new_quantity
            position["total_cost"] = old_cost - cost_basis
            position["avg_price"] = position["total_cost"] / new_quantity if new_quantity > 0 else 0

            # Update capital
            self.current_capital += total_cost

            # Remove position if empty
            if position["quantity"] == 0:
                del self.positions[symbol]

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_value = self.current_capital
        positions_detail = []

        for symbol, position in self.positions.items():
            current_price = self._get_market_price_for_symbol(symbol)

            position_value = current_price * position["quantity"]
            unrealized_pnl = position_value - position["total_cost"]

            positions_detail.append({
                "symbol": symbol,
                "asset_class": position["asset_class"],
                "exchange": position["exchange"],
                "quantity": position["quantity"],
                "avg_price": position["avg_price"],
                "current_price": current_price,
                "position_value": position_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": (unrealized_pnl / position["total_cost"]) * 100 if position["total_cost"] > 0 else 0
            })

            total_value += position_value

        # Calculate total P&L
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "positions": positions_detail,
            "num_positions": len(positions_detail),
            "timestamp": datetime.now()
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"message": "No executions yet"}

        df = pd.DataFrame(self.execution_history)

        # Calculate statistics by asset class
        asset_stats = {}
        for asset_class in df["asset_class"].unique():
            asset_data = df[df["asset_class"] == asset_class]
            asset_stats[asset_class] = {
                "count": len(asset_data),
                "total_volume": asset_data["quantity"].sum(),
                "avg_commission": asset_data["commission"].mean(),
                "total_commission": asset_data["commission"].sum()
            }

        return {
            "total_executions": len(self.execution_history),
            "asset_class_stats": asset_stats,
            "total_commission": df["commission"].sum(),
            "avg_execution_time": "N/A"  # Would track in production
        }

def create_sample_orders() -> List[MultiAssetOrder]:
    """Create sample orders for different asset classes"""
    orders = []

    # Equity order
    equity_contract = ContractSpecification(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        exchange="NASDAQ",
        currency="USD",
        min_tick=0.01
    )

    orders.append(MultiAssetOrder(
        order_id="EQ_001",
        contract=equity_contract,
        side="BUY",
        quantity=100,
        order_type=OrderType.MARKET
    ))

    # Futures order
    futures_contract = ContractSpecification(
        symbol="ES",
        asset_class=AssetClass.FUTURES,
        exchange="CME",
        currency="USD",
        multiplier=50.0,
        min_tick=0.25,
        expiration_date=datetime(2024, 12, 31),
        margin_requirement=12000.0
    )

    orders.append(MultiAssetOrder(
        order_id="FUT_001",
        contract=futures_contract,
        side="BUY",
        quantity=2,
        order_type=OrderType.LIMIT,
        price=4500.0
    ))

    # Options order
    options_contract = ContractSpecification(
        symbol="AAPL_CALL_150",
        asset_class=AssetClass.OPTIONS,
        exchange="CBOE",
        currency="USD",
        multiplier=100.0,
        min_tick=0.01,
        strike_price=150.0,
        option_type="CALL",
        underlying="AAPL",
        expiration_date=datetime(2024, 12, 20)
    )

    orders.append(MultiAssetOrder(
        order_id="OPT_001",
        contract=options_contract,
        side="BUY",
        quantity=10,
        order_type=OrderType.LIMIT,
        price=5.0
    ))

    # Forex order
    forex_contract = ContractSpecification(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        exchange="FOREX",
        currency="USD",
        multiplier=100000.0,
        min_tick=0.00001
    )

    orders.append(MultiAssetOrder(
        order_id="FX_001",
        contract=forex_contract,
        side="BUY",
        quantity=0.1,  # 0.1 lot
        order_type=OrderType.MARKET
    ))

    return orders

def run_multi_asset_demo():
    """Run multi-asset trading demonstration"""
    logging.basicConfig(level=logging.INFO)

    # Create execution engine
    engine = MultiAssetExecutionEngine(initial_capital=1000000.0)

    # Create sample orders
    orders = create_sample_orders()

    # Execute orders
    for order in orders:
        print(f"\nExecuting order: {order.order_id}")
        print(f"Asset: {order.contract.symbol} ({order.contract.asset_class.value})")
        print(f"Side: {order.side}, Quantity: {order.quantity}")

        try:
            result = engine.place_order(order)
            print(f"Status: {result['status']}")
            print(f"Execution Price: ${result['execution_price']:.4f}")
            print(f"Commission: ${result['commission']:.4f}")
            print(f"Total Cost: ${result['total_cost']:.2f}")
        except Exception as e:
            print(f"Error: {e}")

    # Get portfolio summary
    portfolio = engine.get_portfolio_summary()
    print(f"\n=== PORTFOLIO SUMMARY ===")
    print(f"Total Value: ${portfolio['total_value']:,.2f}")
    print(f"Total P&L: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:.2f}%)")
    print(f"Number of Positions: {portfolio['num_positions']}")

    for pos in portfolio['positions']:
        print(f"\n{pos['symbol']} ({pos['asset_class']})")
        print(f"  Quantity: {pos['quantity']}")
        print(f"  Avg Price: ${pos['avg_price']:.4f}")
        print(f"  Current Price: ${pos['current_price']:.4f}")
        print(f"  P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.2f}%)")

    # Get execution statistics
    stats = engine.get_execution_stats()
    print(f"\n=== EXECUTION STATISTICS ===")
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Total Commission: ${stats['total_commission']:.2f}")

    for asset_class, asset_stats in stats['asset_class_stats'].items():
        print(f"\n{asset_class}:")
        print(f"  Count: {asset_stats['count']}")
        print(f"  Volume: {asset_stats['total_volume']}")
        print(f"  Commission: ${asset_stats['total_commission']:.2f}")

if __name__ == "__main__":
    run_multi_asset_demo()
