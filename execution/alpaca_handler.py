# execution/alpaca_handler.py
import os
import requests
import pandas as pd
from typing import List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AlpacaExecutionHandler:
    """
    Live execution handler using Alpaca Markets API.
    FREE paper trading with unlimited virtual capital.
    
    Setup same as AlpacaDataProvider.
    """
    
    def __init__(self, paper=True):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if paper:
            self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        
        if not self.api_key:
            logger.warning("Alpaca execution disabled: API keys not configured")
    
    def submit_order(self, ticker: str, quantity: float, order_type: str = "market"):
        """
        Submit order to Alpaca.
        quantity > 0: BUY
        quantity < 0: SELL
        """
        if not self.api_key:
            logger.error("Cannot submit order: Alpaca not configured")
            return None
        
        try:
            side = "buy" if quantity > 0 else "sell"
            qty = abs(quantity)
            
            # Alpaca requires integer shares for stocks
            qty = int(qty)
            if qty == 0:
                logger.warning(f"Order quantity rounded to zero for {ticker}")
                return None
            
            payload = {
                "symbol": ticker,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": "day"
            }
            
            url = f"{self.base_url}/v2/orders"
            response = requests.post(url, headers=self.headers, json=payload, timeout=10)
            response.raise_for_status()
            
            order_data = response.json()
            logger.info(f"Order submitted: {side} {qty} {ticker}, ID: {order_data.get('id')}")
            return order_data
            
        except Exception as e:
            logger.error(f"Failed to submit order for {ticker}: {e}")
            return None
    
    def get_account(self):
        """Get account details (cash, equity, buying power)."""
        try:
            url = f"{self.base_url}/v2/account"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return None
    
    def get_positions(self):
        """Get current positions."""
        try:
            url = f"{self.base_url}/v2/positions"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_orders(self, status="all"):
        """Get orders (open, closed, all)."""
        try:
            url = f"{self.base_url}/v2/orders"
            params = {"status": status, "limit": 500}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            url = f"{self.base_url}/v2/orders"
            response = requests.delete(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return False
