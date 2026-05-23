import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class TradeSimulator:
    """Lightweight market simulator for paper and dry-run execution."""

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.cash = float(initial_cash)
        self.positions: Dict[str, Dict[str, float]] = {}
        self.order_history: List[Dict[str, Any]] = []

    def execute_market_order(self, symbol: str, qty: float, price: float, side: str, commission: float = 0.0005) -> Dict[str, Any]:
        filled_qty = abs(qty)
        notional = filled_qty * price
        fee = notional * commission

        if side == "buy":
            self.cash -= notional + fee
            self._update_position(symbol, filled_qty, price)
        else:
            self.cash += notional - fee
            self._update_position(symbol, -filled_qty, price)

        result: Dict[str, Any] = {
            "symbol": symbol,
            "qty": filled_qty,
            "side": side,
            "price": price,
            "fee": fee,
            "cash": self.cash
        }
        self.order_history.append(result)
        return result

    def execute_limit_order(
        self,
        symbol: str,
        qty: float,
        limit_price: float,
        side: str,
        commission: float = 0.0005
    ) -> Dict[str, Any]:
        # Simulated limit orders are immediately filled if the limit is at or better than the current price.
        current_price = self._get_simulated_price(symbol)
        if (side == "buy" and limit_price >= current_price) or (side == "sell" and limit_price <= current_price):
            return self.execute_market_order(symbol, qty, limit_price, side, commission)
        return {
            "symbol": symbol,
            "qty": abs(qty),
            "side": side,
            "price": limit_price,
            "status": "pending",
            "fee": 0.0,
            "cash": self.cash
        }

    def _get_simulated_price(self, symbol: str) -> float:
        # Simplified mocked price for limit order evaluation.
        return self.positions.get(symbol, {}).get("avg_price", 100.0)

    def _update_position(self, symbol: str, qty: float, price: float) -> None:
        if symbol not in self.positions:
            self.positions[symbol] = {"qty": 0.0, "avg_price": 0.0}

        current = self.positions[symbol]
        new_qty = current["qty"] + qty
        if new_qty == 0:
            self.positions.pop(symbol, None)
            return

        if qty > 0:
            total_cost = current["avg_price"] * current["qty"] + price * qty
            current["qty"] = new_qty
            current["avg_price"] = total_cost / new_qty
        else:
            current["qty"] = new_qty

        self.positions[symbol] = current

    def close_position(self, symbol: str, price: float) -> Dict[str, Any]:
        position = self.positions.pop(symbol, {"qty": 0.0, "avg_price": 0.0})
        qty = position.get("qty", 0.0)
        if qty == 0:
            return {"symbol": symbol, "qty": 0.0, "price": price, "side": "none", "cash": self.cash}

        side = "sell" if qty > 0 else "buy"
        return self.execute_market_order(symbol, abs(qty), price, side)

    def get_positions(self) -> List[Dict[str, Any]]:
        return [
            {"symbol": symbol, "qty": float(values["qty"]), "avg_price": float(values["avg_price"])}
            for symbol, values in self.positions.items()
        ]

    def get_account(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        market_value = sum(qty["qty"] * current_prices.get(symbol, 0.0) for symbol, qty in self.positions.items())
        total_value = self.cash + market_value
        return {
            "cash": self.cash,
            "portfolio_value": total_value,
            "equity": total_value,
            "buying_power": self.cash,
            "status": "PAPER"
        }
