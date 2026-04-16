
import os
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all AI agents in the hedge fund.
    Provides shared logic for LLM interaction and data access.
    """
    def __init__(self, name: str, model_name: str = "gpt-4-turbo"):
        self.name = name
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY") # Or other provider
        
    @abstractmethod
    def analyze(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary method for agent analysis.
        Should return a standardized dictionary with signal and reasoning.
        """
        pass

    def _call_llm(self, prompt: str) -> str:
        """
        Wrapper for LLM calls. 
        In a production system, this would handle retries and token management.
        For now, we implement a robust interface.
        """
        if not self.api_key:
            logger.warning(f"[{self.name}] No API key found. Returning mock analysis.")
            return "Mock analysis: Signal neutral due to missing API configuration."
            
        # Implementation of actual LLM call would go here
        # (e.g., using litellm or direct SDK)
        return "LLM Analysis response placeholder."

    def standardize_signal(self, raw_signal: str) -> float:
        """Converts strings like 'bullish/buy' to 1.0, 'bearish/sell' to -1.0."""
        s = raw_signal.lower()
        if "buy" in s or "bullish" in s: return 1.0
        if "sell" in s or "bearish" in s: return -1.0
        return 0.0
