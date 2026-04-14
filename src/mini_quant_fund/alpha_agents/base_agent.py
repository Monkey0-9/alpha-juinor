
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import pandas as pd
from contracts import AgentResult

class BaseAgent(ABC):
    """
    Abstract Base Class for all AI Agents.
    Enforces the 'evaluate' contract.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        """
        Produce a forecast for the symbol.
        Must return AgentResult.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": "1.0",
            "type": "alpha"
        }
