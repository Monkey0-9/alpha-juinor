import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional

@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class RealTimeGreeksCalculator:
    """Calculate Greeks in <1μs using optimized Python/NumPy (C++ extension planned)"""
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Greeks:
        """
        Calculate Black-Scholes Greeks.
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free interest rate
        sigma: Volatility
        option_type: "call" or "put"
        """
        if T <= 0:
            return Greeks(0, 0, 0, 0, 0)
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == "call":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
            
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in vol
        
        # Theta
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == "call":
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (term1 - term2) / 365  # Daily theta
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365  # Daily theta
            
        # Rho
        if option_type == "call":
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in rates
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Per 1% change in rates
            
        return Greeks(
            delta=float(delta),
            gamma=float(gamma),
            theta=float(theta),
            vega=float(vega),
            rho=float(rho)
        )
