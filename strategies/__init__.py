"""
Institutional Trading Strategies Package.

This package contains various trading strategies including:
- Alpha generation strategies
- Risk management strategies
- Portfolio allocation strategies
- Statistical arbitrage strategies
"""

# Import key classes for convenience
from .stat_arb import StatArbEngine
from .alpha import CompositeAlpha, TrendAlpha, MeanReversionAlpha
from .composite_alpha import CompositeAlphaStrategy
from .factory import StrategyFactory
from .base import BaseStrategy
from .institutional_strategy import InstitutionalStrategy
# Note: MLAlpha is in ml_models subpackage, import via:
# from strategies.ml_models.ml_alpha import MLAlpha

__all__ = [
    'StatArbEngine',
    'CompositeAlpha',
    'TrendAlpha',
    'MeanReversionAlpha',
    'CompositeAlphaStrategy',
    'StrategyFactory',
    'BaseStrategy',
    'InstitutionalStrategy',
]

