"""
Free Broker Integration - Alpaca Free Tier and Paper Trading
"""

import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

class AlpacaFreeBroker:
    """Free Alpaca broker integration (paper trading)"""

    def __init__(self, paper_trading: bool = True):
        """
        Initialize Alpaca broker

        Args:
            paper_trading: Use paper trading (free)
        """
        self.paper_trading = paper_trading
        self.api_key = "YOUR_ALPACA_API_KEY"  # Free API key from Alpaca
        self.api_secret = "YOUR_ALPACA_SECRET_KEY"

        if paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"

        self.session = requests.Session()
        if self.api_key != "YOUR_ALPACA_API_KEY":
            self.session.headers.update({
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.api_secret
            })

    def get_account(self) -> Dict:
        """
        Get account information

        Returns:
            Account details
        """
        try:
            if self.api_key == "YOUR_ALPACA_API_KEY":
                # Mock account data for demo
                return {
                    "account_id": "882190-ELITE",
                    "buying_power": 20000000.0,
                    "cash": 10000000.0,
                    "portfolio_value": 15000000.0,
                    "equity": 15000000.0,
                    "long_market_value": 12000000.0,
                    "short_market_value": 3000000.0,
                    "initial_margin": 6000000.0,
                    "maintenance_margin": 3000000.0,
                    "daytrading_buying_power": 40000000.0,
                    "regt_buying_power": 20000000.0,
                    "status": "ACTIVE",
                    "source": "Alpaca_Mock"
                }

            response = self.session.get(f"{self.base_url}/v2/account")
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error fetching account: {e}")
            return {"error": str(e)}

    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "market",
                   time_in_force: str = "day", limit_price: float = None) -> Dict:
        """
        Place an order

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders

        Returns:
            Order confirmation
        """
        try:
            if self.api_key == "YOUR_ALPACA_API_KEY":
                # Mock order response
                order_id = f"order_{int(time.time())}"
                return {
                    "order_id": order_id,
                    "client_order_id": f"client_{order_id}",
                    "symbol": symbol,
                    "qty": qty,
                    "side": side,
                    "order_type": order_type,
                    "time_in_force": time_in_force,
                    "limit_price": limit_price,
                    "status": "accepted",
                    "created_at": datetime.now().isoformat(),
                    "source": "Alpaca_Mock"
                }

            order_data = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force
            }

            if limit_price and order_type in ["limit", "stop_limit"]:
                order_data["limit_price"] = limit_price

            response = self.session.post(f"{self.base_url}/v2/orders", json=order_data)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"error": str(e)}

    def get_positions(self) -> List[Dict]:
        """
        Get current positions

        Returns:
            List of positions
        """
        try:
            if self.api_key == "YOUR_ALPACA_API_KEY":
                # Mock positions
                return [
                    {
                        "symbol": "AAPL",
                        "qty": 1000,
                        "market_value": 150000.0,
                        "cost_basis": 140000.0,
                        "unrealized_pl": 10000.0,
                        "unrealized_plpc": 7.14,
                        "side": "long"
                    },
                    {
                        "symbol": "MSFT",
                        "qty": 500,
                        "market_value": 125000.0,
                        "cost_basis": 130000.0,
                        "unrealized_pl": -5000.0,
                        "unrealized_plpc": -3.85,
                        "side": "long"
                    }
                ]

            response = self.session.get(f"{self.base_url}/v2/positions")
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def get_orders(self, status: str = "open") -> List[Dict]:
        """
        Get orders

        Args:
            status: 'open', 'closed', 'all'

        Returns:
            List of orders
        """
        try:
            if self.api_key == "YOUR_ALPACA_API_KEY":
                # Mock orders
                return [
                    {
                        "order_id": "order_12345",
                        "symbol": "AAPL",
                        "qty": 100,
                        "side": "buy",
                        "order_type": "market",
                        "status": "filled",
                        "filled_qty": 100,
                        "filled_avg_price": 150.25
                    }
                ]

            params = {"status": status}
            response = self.session.get(f"{self.base_url}/v2/orders", params=params)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []

class FreePaperTradingSimulator:
    """Free paper trading simulator for testing"""

    def __init__(self, initial_capital: float = 1000000.0):
        """
        Initialize paper trading simulator

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.commission_rate = 0.001  # 0.1% commission

    def place_order(self, symbol: str, qty: int, side: str, price: float = None) -> Dict:
        """
        Place a paper trading order

        Args:
            symbol: Stock symbol
            qty: Quantity
            side: 'buy' or 'sell'
            price: Price (if None, uses market price)

        Returns:
            Order confirmation
        """
        try:
            # Get market price if not provided
            if price is None:
                price = self._get_market_price(symbol)

            # Calculate commission
            commission = abs(qty * price * self.commission_rate)

            # Check if sufficient funds for buy orders
            if side == "buy":
                required_capital = qty * price + commission
                if required_capital > self.current_capital:
                    return {"error": "Insufficient funds"}

            # Execute order
            order_id = f"paper_{int(time.time())}"

            if side == "buy":
                # Add to positions
                if symbol in self.positions:
                    self.positions[symbol]["qty"] += qty
                    self.positions[symbol]["avg_price"] = (
                        (self.positions[symbol]["avg_price"] * self.positions[symbol]["qty"] + price * qty) /
                        (self.positions[symbol]["qty"] + qty)
                    )
                else:
                    self.positions[symbol] = {
                        "qty": qty,
                        "avg_price": price,
                        "total_cost": qty * price
                    }

                self.current_capital -= required_capital

            else:  # sell
                if symbol not in self.positions or self.positions[symbol]["qty"] < qty:
                    return {"error": "Insufficient position"}

                # Update position
                self.positions[symbol]["qty"] -= qty
                proceeds = qty * price - commission
                self.current_capital += proceeds

                # Remove position if quantity is zero
                if self.positions[symbol]["qty"] == 0:
                    del self.positions[symbol]

            # Record trade
            trade = {
                "timestamp": datetime.now().isoformat(),
                "order_id": order_id,
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "price": price,
                "commission": commission,
                "total_cost": qty * price + commission
            }

            self.trade_history.append(trade)

            return {
                "order_id": order_id,
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "price": price,
                "commission": commission,
                "status": "filled",
                "timestamp": trade["timestamp"]
            }

        except Exception as e:
            logger.error(f"Error placing paper order: {e}")
            return {"error": str(e)}

    def get_account(self) -> Dict:
        """
        Get account information

        Returns:
            Account details
        """
        try:
            return {
                "account_id": "FREE_PAPER_TRADING",
                "buying_power": self.current_capital,
                "cash": self.current_capital,
                "portfolio_value": self.current_capital,
                "equity": self.current_capital,
                "status": "ACTIVE",
                "source": "Free_Paper_Trading"
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {"error": str(e)}

    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary

        Returns:
            Portfolio summary
        """
        try:
            total_value = self.current_capital
            positions_value = 0

            for symbol, position in self.positions.items():
                current_price = self._get_market_price(symbol)
                position_value = position["qty"] * current_price
                positions_value += position_value
                position["current_value"] = position_value
                position["unrealized_pl"] = position_value - position["total_cost"]

            total_value += positions_value
            total_return = (total_value - self.initial_capital) / self.initial_capital

            return {
                "initial_capital": self.initial_capital,
                "current_capital": self.current_capital,
                "positions_value": positions_value,
                "total_value": total_value,
                "total_return": total_return,
                "total_return_pct": total_return * 100,
                "positions": self.positions,
                "num_trades": len(self.trade_history),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {"error": str(e)}

    def _get_market_price(self, symbol: str) -> float:
        """
        Get market price for a symbol (mock implementation)

        Args:
            symbol: Stock symbol

        Returns:
            Market price
        """
        # Mock market prices with some randomness
        base_prices = {
            "AAPL": 150.0,
            "MSFT": 250.0,
            "GOOGL": 2800.0,
            "AMZN": 3200.0,
            "TSLA": 800.0,
            "WMT": 140.0,
            "TGT": 200.0,
            "COST": 450.0,
            "HD": 300.0,
            "LOW": 200.0
        }

        base_price = base_prices.get(symbol, 100.0)
        # Add some random variation
        price = base_price * (1 + np.random.normal(0, 0.02))

        return price

def get_free_broker_integration(broker_type: str = "alpaca") -> Dict:
    """
    Get free broker integration

    Args:
        broker_type: Type of broker ('alpaca', 'paper')

    Returns:
        Broker integration
    """
    try:
        if broker_type == "alpaca":
            return {"broker": AlpacaFreeBroker(paper_trading=True)}
        elif broker_type == "paper":
            return {"broker": FreePaperTradingSimulator()}
        else:
            return {"error": "Invalid broker type"}

    except Exception as e:
        logger.error(f"Error setting up free broker: {e}")
        return {"error": str(e)}
