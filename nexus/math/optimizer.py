"""
nexus/math/optimizer.py — Legacy Alias

Consolidated into nexus.math.optimization.
This module re-exports PortfolioOptimizer for backward compatibility.
"""
from nexus.math.optimization import PortfolioOptimizer, KellyCriterionSizer, InformationCoefficientTracker

__all__ = ["PortfolioOptimizer", "KellyCriterionSizer", "InformationCoefficientTracker"]
