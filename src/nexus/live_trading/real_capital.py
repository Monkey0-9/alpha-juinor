
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RealCapitalManager:
    """
    Manages actual trading capital, enforcing strict risk limits and tracking real-time P&L.
    """
    
    def __init__(self, initial_capital: float, max_drawdown_pct: float = 0.1):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_capital = initial_capital
        self.pnl_realized = 0.0
        self.pnl_unrealized = 0.0
        self.is_trading = False
        self.last_update = datetime.now()
        
    def start_trading(self):
        """Enable live trading with real capital."""
        if self.current_capital <= 0:
            logger.error("Cannot start trading with zero or negative capital.")
            return
        
        logger.warning(f"!!! STARTING LIVE TRADING WITH ${self.current_capital:,.2f} !!!")
        self.is_trading = True
        
    def stop_trading(self, reason: str = "Manual stop"):
        """Disable live trading."""
        logger.info(f"STOPPING LIVE TRADING. Reason: {reason}. Final Capital: ${self.current_capital:,.2f}")
        self.is_trading = False
        
    def update_pnl(self, realized: float, unrealized: float):
        """Update capital based on P&L."""
        self.pnl_realized += realized
        self.pnl_unrealized = unrealized
        self.current_capital = self.initial_capital + self.pnl_realized + self.pnl_unrealized
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            
        self.last_update = datetime.now()
        
        if not self.check_risk_limits():
            self.stop_trading("Risk limit violation (Max Drawdown)")

    def check_risk_limits(self) -> bool:
        """
        Enforce drawdown limits.
        Returns False if limits are breached.
        """
        if self.peak_capital <= 0:
            return True
            
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        if current_drawdown >= self.max_drawdown_pct:
            logger.critical(f"MAX DRAWDOWN BREACHED: {current_drawdown:.2%} >= {self.max_drawdown_pct:.2%}")
            return False
            
        return True

    def get_status(self) -> Dict:
        """Return current capital status."""
        return {
            "is_trading": self.is_trading,
            "current_capital": self.current_capital,
            "total_pnl": self.pnl_realized + self.pnl_unrealized,
            "drawdown": (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0,
            "last_update": self.last_update.isoformat()
        }
