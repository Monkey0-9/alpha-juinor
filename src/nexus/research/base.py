from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from ..models.market import MarketBar
from ..models.trade import OrderSide

class Signal(BaseModel):
    """
    Normalized alpha signal output.
    """
    symbol: str
    timestamp: datetime
    value: float = Field(..., description="Signal value, typically normalized [-1, 1]")
    side: Optional[OrderSide] = None
    confidence: float = 0.0
    reason_code: str = "UNDEFINED"
    metadata: Dict[str, Any] = {}

class BaseAlpha(ABC):
    """
    Abstract base class for all Alpha strategies.
    Enforces a strict interface for signal generation and evaluation.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, data: List[MarketBar]) -> Optional[Signal]:
        """
        Compute signal based on input market bars.
        Must strictly avoid look-ahead bias.
        """
        pass

    def get_name(self) -> str:
        return self.name
