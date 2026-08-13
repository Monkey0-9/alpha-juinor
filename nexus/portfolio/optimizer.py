"""
nexus/math/optimizer.py — Legacy Alias

Consolidated into nexus.portfolio.optimization.
This module re-exports PortfolioOptimizer for backward compatibility.
"""

from nexus.portfolio.optimization import (
    PortfolioOptimizer,
    KellyCriterionSizer,
    InformationCoefficientTracker,
)

__all__ = [
    "PortfolioOptimizer",
    "KellyCriterionSizer",
    "InformationCoefficientTracker",
]
