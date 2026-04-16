#!/usr/bin/env python3
"""
Social Copy Trading Manager
===========================
Logic for managing follower portfolios and mirroring lead trader actions.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class CopyTradingManager:
    """
    Manages social copy trading functionality.
    Allows followers to automatically mirror the trades of a lead trader
    with proportional position sizing and risk limits.
    """
    
    def __init__(self, lead_trader_id: str):
        self.lead_trader_id = lead_trader_id
        self.followers: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
    def add_follower(self, follower_id: str, capital_allocation: float, risk_multiplier: float = 1.0):
        """
        Registers a new follower to the lead trader.
        
        Args:
            follower_id: Unique identifier for the follower.
            capital_allocation: Total capital the follower has dedicated to copy trading.
            risk_multiplier: Multiplier for lead trader's position sizes (default 1.0).
        """
        self.followers[follower_id] = {
            "id": follower_id,
            "capital": capital_allocation,
            "risk_multiplier": risk_multiplier,
            "active": True,
            "joined_at": datetime.utcnow()
        }
        logger.info(f"Follower {follower_id} added to lead trader {self.lead_trader_id}")

    def remove_follower(self, follower_id: str):
        """Unregisters a follower."""
        if follower_id in self.followers:
            del self.followers[follower_id]
            logger.info(f"Follower {follower_id} removed")

    def mirror_trade(self, symbol: str, side: str, lead_quantity: int, lead_price: float):
        """
        Mirrors a lead trader's trade to all active followers.
        
        Logic:
        Follower Quantity = (Lead Trade Value / Lead Portfolio Value) * Follower Portfolio Value * Risk Multiplier
        
        Args:
            symbol: Ticker symbol
            side: 'buy' or 'sell'
            lead_quantity: Number of shares/contracts traded by lead
            lead_price: Execution price
        """
        # Mock lead portfolio value for calculation
        LEAD_PORTFOLIO_VALUE = 10_000_000.0 
        
        trade_event = {
            "symbol": symbol,
            "side": side,
            "lead_qty": lead_quantity,
            "price": lead_price,
            "timestamp": datetime.utcnow(),
            "follower_executions": []
        }

        for fid, config in self.followers.items():
            if not config["active"]:
                continue
                
            # Proportional sizing
            proportion = (lead_quantity * lead_price) / LEAD_PORTFOLIO_VALUE
            follower_qty = int((proportion * config["capital"] * config["risk_multiplier"]) / lead_price)
            
            if follower_qty > 0:
                execution = {
                    "follower_id": fid,
                    "qty": follower_qty,
                    "status": "mirrored"
                }
                trade_event["follower_executions"].append(execution)
                logger.debug(f"Mirrored {side} {follower_qty} {symbol} for follower {fid}")

        self.trade_history.append(trade_event)
        return trade_event

    def get_stats(self):
        """Returns statistics for the copy trading group."""
        return {
            "lead_trader": self.lead_trader_id,
            "total_followers": len(self.followers),
            "active_followers": len([f for f in self.followers.values() if f["active"]]),
            "total_trades_mirrored": len(self.trade_history)
        }

if __name__ == "__main__":
    # Example usage
    manager = CopyTradingManager("EliteQuant_01")
    manager.add_follower("User_42", capital_allocation=100000.0)
    manager.add_follower("User_99", capital_allocation=50000.0, risk_multiplier=2.0)
    
    result = manager.mirror_trade("AAPL", "buy", 5000, 180.0)
    print(f"Trade Mirrored: {result}")
    print(f"Stats: {manager.get_stats()}")
