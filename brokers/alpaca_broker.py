
import requests
import logging
import json
import time
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
        
        # Verify connection
        try:
            self.account = self.get_account()
            logger.info(f"Connected to Alpaca: {self.account.get('status')} | Equity: ${self.account.get('equity')}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise

    def get_account(self) -> Dict:
        """Get account details."""
        r = self.session.get(f"{self.base_url}/v2/account")
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> Dict[str, float]:
        """Get current positions as {ticker: quantity}."""
        r = self.session.get(f"{self.base_url}/v2/positions")
        r.raise_for_status()
        data = r.json()
        positions = {}
        for pos in data:
            positions[pos['symbol']] = float(pos['qty'])
        return positions

    def submit_orders(self, orders: List[Order]) -> List[Dict]:
        """
        Execute a list of orders.
        Does not support atomic batching natively, uses loop.
        """
        results = []
        for order in orders:
            side = "buy" if order.quantity > 0 else "sell"
            qty = abs(order.quantity)
            if qty < 0.0001: 
                continue # ignore tiny noise
            
            # Alpaca requires string side
            payload = {
                "symbol": order.ticker,
                "qty": str(qty),  # fractional shares supported
                "side": side,
                "type": "market", # simplification for now
                "time_in_force": "day"
            }
            
            try:
                r = self.session.post(f"{self.base_url}/v2/orders", json=payload)
                if r.status_code in [200, 201]:
                    logger.info(f"Submitted {side} {qty} {order.ticker}")
                    results.append({"status": "submitted", "order": r.json()})
                else:
                    logger.error(f"Order failed {side} {order.ticker}: {r.text}")
                    results.append({"status": "failed", "error": r.text})
            except Exception as e:
                logger.error(f"Exception submitting {order.ticker}: {e}")
                results.append({"status": "error", "error": str(e)})
                
        return results

    def close_all(self):
        """Liquidate all positions (Panic button)."""
        r = self.session.delete(f"{self.base_url}/v2/positions")
        return r.status_code == 207 # Multi-status
