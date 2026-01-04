
import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class RiskAgent(BaseAgent):
    """
    Specialized agent for Qualitative Risk Management.
    Analyzes sector concentration and volatility risks.
    """
    def __init__(self, model_name: str = "gpt-4-turbo"):
        super().__init__("RiskManager", model_name)

    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the risk of adding a position.
        """
        prices = data.get("prices")
        
        if prices is not None and not prices.empty:
            vol = prices['Close'].pct_change().std() * (252**0.5)
        else:
            vol = 0.3 # Default estimate
            
        risk_limit = 0.4 # 40% annual vol limit for full size
        
        signal_scale = 1.0
        if vol > risk_limit:
            signal_scale = risk_limit / vol # Scale down signal for high vol
            
        return {
            "ticker": ticker,
            "signal": signal_scale, # In RiskAgent, signal represents a multiplier (0 to 1)
            "confidence": 0.9,
            "reason": f"Annualized volatility is {vol:.2%}. Risk scaling set to {signal_scale:.2f}x.",
            "agent": self.name
        }
