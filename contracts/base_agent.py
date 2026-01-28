
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd

@dataclass
class AgentResult:
    symbol: str
    agent_name: str
    mu: float
    sigma: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional conversion helper
    @property
    def score(self) -> float:
        """Normalized signal score often treated as mu."""
        return self.mu

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "agent_name": self.agent_name,
            "mu": self.mu,
            "sigma": self.sigma,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, symbol: str, data: pd.DataFrame) -> AgentResult:
        """
        Evaluate the agent on the given data.
        Returns an AgentResult.
        """
        pass
