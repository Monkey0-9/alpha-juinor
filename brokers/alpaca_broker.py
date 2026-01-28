
import requests
import logging
import json
import time
import uuid
import math
from typing import List, Dict, Optional, Any, Union
from backtest.execution import Order, OrderType

logger = logging.getLogger(__name__)

class AlpacaExecutionHandler:
    """
    Handles execution against Alpaca Markets (Paper/Live).
    Uses direct REST API via requests to avoid SDK dependency conflicts.
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.base_url = base_url
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        try:
            self.account = self.get_account()
            logger.info(f"Connected to Alpaca: {self.account.get('status')} | Equity: ${self.account.get('equity')}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Internal retry wrapper for Alpaca API."""
        retries = kwargs.pop('retries', 3)
        for i in range(retries):
            try:
                response = self.session.request(method, url, timeout=15, **kwargs)
                if response.status_code >= 400:
                    logger.error(f"Alpaca API Error {response.status_code}: {response.text}")
                response.raise_for_status()
                return response
            except Exception as e:
                if i == retries - 1:
                    raise
                time.sleep(1)
        return None

    def get_account(self) -> Dict:
        """Get account details."""
        r = self._request_with_retry("GET", f"{self.base_url}/v2/account")
        return r.json()

    def get_positions(self) -> Dict[str, float]:
        """Get current positions as {ticker: quantity}."""
        r = self._request_with_retry("GET", f"{self.base_url}/v2/positions")
        data = r.json()
        positions = {}
        for pos in data:
            positions[pos['symbol']] = float(pos['qty'])
        return positions

    def submit_orders(self, orders: List[Order]) -> List[Dict]:
        """
        Execute a list of orders.
        Includes idempotency keys and fractional rounding (4 decimals).
        """
        results = []
        for order in orders:
            # Alpaca Ticker Mapping
            ticker = order.ticker.replace("-", "") if "-" in order.ticker else order.ticker

            side = "buy" if order.quantity > 0 else "sell"
            qty = abs(order.quantity)
            if qty < 0.0001:
                continue

            # IDEMPOTENCY: Use UUID for client_order_id to prevent double-fills on retry
            client_order_id = str(uuid.uuid4())

            payload = {
                "symbol": ticker,
                "qty": str(round(qty, 4)),
                "side": side,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": client_order_id
            }

            try:
                # Use retry wrapper for POST
                r = self._request_with_retry("POST", f"{self.base_url}/v2/orders", json=payload)
                order_data = r.json()

                reason_log = f"REASON: {order.reason}"
                if order.risk_metric_triggered:
                    reason_log += f" | RISK: {order.risk_metric_triggered}"

                logger.info(f"Submitted {side} {qty:.4f} {order.ticker} | {reason_log} | ID: {order_data.get('id')} | CID: {client_order_id}")
                results.append({"status": "submitted", "order": order_data})
            except Exception as e:
                logger.error(f"Failed to submit {order.ticker} ({order.reason}): {e}")
                results.append({"status": "failed", "error": str(e)})

        return results

    def get_orders(self, status: str = "open", limit: int = 50) -> List[Dict]:
        """Fetch historical or open orders."""
        params = {"status": status, "limit": limit}
        r = self.session.get(f"{self.base_url}/v2/orders", params=params)
        r.raise_for_status()
        return r.json()

    def get_activities(self, type: str = "FILL") -> List[Dict]:
        """Fetch account activities (e.g. FILLS, CANCELS)."""
        r = self.session.get(f"{self.base_url}/v2/account/activities/{type}")
        r.raise_for_status()
        return r.json()

    def close_all(self):
        """Liquidate all positions (Panic button)."""
        r = self.session.delete(f"{self.base_url}/v2/positions")
        return r.status_code == 207 # Multi-status

    def cancel_all_orders(self):
        """Cancel all open orders."""
        r = self.session.delete(f"{self.base_url}/v2/orders")
        return r.status_code == 200 or r.status_code == 207

    def _safe_get_account(self) -> Optional[Dict]:
        """Return account object or None on failure."""
        try:
            return self.get_account()
        except Exception as e:
            logger.error("Failed to fetch Alpaca account: %s", e)
            return None

    def submit_order(self, symbol: str, qty: float, side: str, type: str = "market", time_in_force: str = "day", limit_price: float = None, stop_price: float = None, price: float = None) -> Dict[str, Any]:
        """
        Robust submit: checks buying power and disallows fractional short sells.
        Price arg is optional; used for buying power calculation.
        """
        try:
            # 1. Normalize Symbol
            ticker = symbol.replace("-", "") if "-" in symbol and "USD" not in symbol else symbol
            if "USD" in ticker and "-" in ticker:
                ticker = ticker.replace("-", "/")

            # 2. Prevent Fractional Short Sells (Round down SELL)
            is_sell = side.lower() == "sell"
            if is_sell:
                # Alpaca 422: fractional short sales not allowed
                int_qty = int(math.floor(float(qty)))
                if int_qty <= 0:
                    return {
                        "success": False,
                        "order": None,
                        "error": "QTY_ROUNDED_TO_ZERO_FOR_SELL",
                        "mapped_symbol": ticker
                    }
                qty_to_send = int_qty
            else:
                qty_to_send = float(qty)

            if qty_to_send <= 0:
                return {
                    "success": False,
                    "order": None,
                    "error": "Qty must be positive",
                    "mapped_symbol": ticker
                }

            # 3. Check Account / Buying Power
            acct = self._safe_get_account()
            if acct is None:
                return {"success": False, "order": None, "error": "ACCOUNT_FETCH_FAILED", "mapped_symbol": ticker}

            # Calculate Notional Required for BUYS
            if side.lower() == "buy":
                order_price = float(price) if price else None

                if order_price:
                    notional_required = order_price * qty_to_send
                    buying_power = float(acct.get('buying_power', 0))
                    safety_buffer = 5.0 # USD

                    if notional_required + safety_buffer > buying_power:
                        return {
                            "success": False,
                            "order": None,
                            "error": "INSUFFICIENT_BUYING_POWER",
                            "detail": {"buying_power": buying_power, "required": notional_required + safety_buffer},
                            "mapped_symbol": ticker
                        }
                else:
                    logger.warning(f"No price provided for BP check for {ticker}")


            # 4. Check Shorting Capability for SELLS
            if is_sell:
                 # Check if we have position
                try:
                    resp = self.session.get(f"{self.base_url}/v2/positions/{ticker}")
                    if resp.status_code != 200:
                        # No long position, check margin/shorting enabled
                        shorting_allowed = acct.get('shorting_enabled', False) or float(acct.get('margin_balance', 0)) != 0
                        if not shorting_allowed:
                            return {
                                "success": False,
                                "order": None,
                                "error": "SHORTING_NOT_ALLOWED",
                                "mapped_symbol": ticker
                            }
                except: pass

            # 5. Submit Payload
            payload = {
                "symbol": ticker,
                "qty": str(qty_to_send),
                "side": side.lower(),
                "type": type.lower(),
                "time_in_force": time_in_force.lower(),
                "client_order_id": str(uuid.uuid4())
            }

            if limit_price:
                payload["limit_price"] = str(limit_price)
            if stop_price:
                payload["stop_price"] = str(stop_price)

            # Submit with Retry (3 attempts)
            retries = 3
            backoff = 0.5
            last_exc = None
            for attempt in range(retries):
                try:
                    response = self._request_with_retry("POST", f"{self.base_url}/v2/orders", json=payload)
                    order_data = response.json()
                    logger.info(f"Submitted {side} {qty_to_send} {ticker} | ID: {order_data.get('id')}")
                    return {
                        "success": True,
                        "order": order_data,
                        "error": None,
                        "mapped_symbol": ticker
                    }
                except Exception as exc:
                    last_exc = exc
                    logger.warning(f"Alpaca submission attempt {attempt+1} failed: {exc}")
                    if attempt < retries - 1:
                        time.sleep(backoff * (2 ** attempt))

            return {
                "success": False,
                "order": None,
                "error": str(last_exc),
                "mapped_symbol": ticker
            }

        except Exception as e:
            logger.exception("submit_order unexpected exception")
            return {
                "success": False,
                "order": None,
                "error": str(e),
                "mapped_symbol": symbol
            }
