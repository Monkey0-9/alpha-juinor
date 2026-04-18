"""
Multi-Asset Class Support
========================
Jane Street, Citadel, Jump Trading, Optiver - all operate across multiple asset classes.
This module enables equities, fixed income, crypto, derivatives, and FX trading.
"""

from .manager import AssetClassManager
from .strategies.market_making import MarketMakingEngine
from .strategies.arbitrage import ArbitrageEngine
from .strategies.statistical import StatisticalAlphaEngine

__all__ = [
    "AssetClassManager",
    "MarketMakingEngine",
    "ArbitrageEngine",
    "StatisticalAlphaEngine",
]
