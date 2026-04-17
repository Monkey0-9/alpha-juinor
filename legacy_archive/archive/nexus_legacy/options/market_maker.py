from typing import List, Dict, Tuple
from dataclasses import dataclass
from .greeks_calculator import RealTimeGreeksCalculator, Greeks

@dataclass
class Quote:
    bid: float
    ask: float
    bid_size: int
    ask_size: int

class OptionsMarketMaker:
    """Institutional Two-Sided Quote Engine with Inventory Skew"""
    
    def __init__(self, max_inventory: int = 1000):
        self.greeks_calc = RealTimeGreeksCalculator()
        self.inventory = 0 # Net Delta-equivalent position
        self.max_inventory = max_inventory
        
    def generate_quotes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Quote:
        """
        Generate quotes with fair value, inventory skew, and adverse selection protection.
        """
        # 1. Theoretical Fair Value (BS Model)
        theo = self._calculate_theo(S, K, T, r, sigma, option_type)
        
        # 2. Dynamic Spread (Function of Volatility and Gamma)
        # Wider spreads in high vol / high gamma regimes
        greeks = self.greeks_calc.calculate_greeks(S, K, T, r, sigma, option_type)
        base_spread = theo * (0.005 + 0.1 * greeks.gamma)
        
        # 3. Inventory Skew (Avellaneda-Stoikov Logic)
        # If we are long (inventory > 0), we lower our bid/ask to encourage selling
        skew = (self.inventory / self.max_inventory) * (base_spread * 0.5)
        
        bid = theo - (base_spread / 2) - skew
        ask = theo + (base_spread / 2) - skew
        
        return Quote(
            bid=max(0.01, round(bid, 2)),
            ask=max(0.02, round(ask, 2)),
            bid_size=100,
            ask_size=100
        )
        
    def _calculate_theo(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        from scipy.stats import norm
        import numpy as np
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def _calculate_skew(self, option_type: str, strike: float) -> float:
        # Placeholder for inventory-based skew
        return 0.0
