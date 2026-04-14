
import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class PortfolioAgent(BaseAgent):
    """
    Specialized agent for Portfolio Management.
    Makes the final sizing decision based on all other agent signals.
    """
    def __init__(self, model_name: str = "gpt-4-turbo"):
        super().__init__("PortfolioManager", model_name)

    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combines signals and confidence into a final position size.
        """
        avg_signal = data.get("avg_signal", 0)
        max_position_size = 0.20 # 20% max per ticker
        
        final_size = avg_signal * max_position_size
        
        return {
            "ticker": ticker,
            "signal": final_size, # This is the actual weight in the portfolio (e.g. 0.1 for 10%)
            "confidence": 1.0,
            "reason": f"Sizing capped at {max_position_size:.0%} per asset. Target weight: {final_size:.1%}.",
            "agent": self.name
        }
