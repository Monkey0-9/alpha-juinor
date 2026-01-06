"""
Base Alpha Family Class for Institutional Trading.

Provides common interface and utilities for all alpha families.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseAlpha(ABC):
    """
    Abstract base class for alpha families.

    All alpha families must implement generate_signal method.
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, regime_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate alpha signal.

        Args:
            data: OHLCV data
            regime_context: Current market regime info

        Returns:
            Dict with signal, confidence, and metadata
        """
        if data is None or data.empty or "Close" not in data.columns:
            return {'signal': 0.0, 'confidence': 0.0, 'metadata': {'error': 'Invalid data or missing Close column'}}
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns."""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_cols)

    def normalize_signal(self, signal: float) -> float:
        """Normalize signal to [-1, 1] range."""
        return max(-1.0, min(1.0, signal))
