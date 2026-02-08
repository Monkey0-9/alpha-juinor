"""
Derivatives Module
==================

Advanced options and derivatives pricing, hedging, and analytics.

Modules:
- volatility_surface: Implied volatility surface construction
- exotic_options: Exotic options pricing (barrier, Asian, lookback)
- greeks: Option Greeks calculation (delta, gamma, vega, theta, rho)
"""

from derivatives.exotic_options import (
    AsianOption,
    BarrierOption,
    LookbackOption,
    price_asian_option,
    price_barrier_option,
)
from derivatives.volatility_surface import SABRModel, VolatilitySurface

__all__ = [
    "VolatilitySurface",
    "SABRModel",
    "BarrierOption",
    "AsianOption",
    "LookbackOption",
    "price_barrier_option",
    "price_asian_option",
]
