import numpy as np
from typing import List, Dict

class AllWeatherPortfolio:
    """Risk-parity allocation across regimes (Scaffold)"""
    
    def construct_portfolio(self, volatilities: np.ndarray) -> np.ndarray:
        """
        Risk Parity: weights inversely proportional to volatility.
        """
        inv_vol = 1.0 / volatilities
        return inv_vol / inv_vol.sum()
