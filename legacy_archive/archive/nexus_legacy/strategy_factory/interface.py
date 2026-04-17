from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from typing import Optional

@dataclass
class Signal:
    """Standardized trading signal."""
    symbol: str
    strength: float      # -1.0 to 1.0 (Direction + Conviction)
    confidence: float    # 0.0 to 1.0 (Statistical confidence)
    regime_adjusted: bool
    metadata: dict
    is_entry: bool = False

class StrategyInterface(ABC):
    """
    Abstract base class for all alpha engines.
    Enforces standardized Input/Output for the factory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        prices: pd.Series,
        regime_data: Optional[dict] = None
    ) -> Signal:
        """
        Generate a standardized trading signal.

        Args:
            symbol: Ticker symbol
            prices: Series of historical close prices
            regime_data: Optional dict with keys 'regime', 'risk_mult'

        Returns:
            Signal object
        """
        pass
