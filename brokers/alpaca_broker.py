
import requests
import logging
import json
import time
import uuid
from typing import List, Dict, Optional
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
                if response.status_code == 429: # Rate limit
                    wait = 2 ** i
                    logger.warning(f"Alpaca Rate Limit. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response
            except Exception as e:
                if i == retries - 1:
                    logger.error(f"Alpaca API Error after {retries} retries: {e}")
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
