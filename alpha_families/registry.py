# alpha_families/registry.py
# Single entry point returning instantiated alpha family objects.

from .momentum_ts import MomentumTS
from .mean_reversion import MeanReversionAlpha
from .volatility_carry import VolatilityCarry
from .trend_strength import TrendStrength

def get_alpha_families():
    """
    Returns a list of instantiated alpha family objects.
    Strategies should import only this function:
        from alpha_families import get_alpha_families
    """
    return [
        MomentumTS(),
        MeanReversionAlpha(),
        VolatilityCarry(),
        TrendStrength(),
    ]
