import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Dict
from numba import njit, float64
from datetime import datetime
import math

@njit
def scalar_n_cdf(x):
    """Fast normal CDF approximation for Numba (Scalar)"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@njit
def scalar_n_pdf(x):
    """Fast normal PDF for Numba (Scalar)"""
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)

@njit
def calculate_greeks_scalar(S, K, T, r, sigma, is_call):
    if T <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Delta
    if is_call:
        delta = scalar_n_cdf(d1)
    else:
        delta = scalar_n_cdf(d1) - 1.0
        
    # Gamma
    gamma = scalar_n_pdf(d1) / (S * sigma * math.sqrt(T))
    
    # Vega
    vega = S * scalar_n_pdf(d1) * math.sqrt(T) / 100.0
    
    # Theta
    term1 = -(S * scalar_n_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if is_call:
        term2 = r * K * math.exp(-r * T) * scalar_n_cdf(d2)
        theta = (term1 - term2) / 365.0
    else:
        term2 = r * K * math.exp(-r * T) * scalar_n_cdf(-d2)
        theta = (term1 + term2) / 365.0
        
    # Rho
    if is_call:
        rho = K * T * math.exp(-r * T) * scalar_n_cdf(d2) / 100.0
    else:
        rho = -K * T * math.exp(-r * T) * scalar_n_cdf(-d2) / 100.0
        
    theoretical_price = is_call * (S * scalar_n_cdf(d1) - K * math.exp(-r * T) * scalar_n_cdf(d2)) + \
                        (1 - is_call) * (K * math.exp(-r * T) * scalar_n_cdf(-d2) - S * scalar_n_cdf(-d1))

    delta_gamma = -(d1 * scalar_n_pdf(d1)) / (S * S * sigma * sigma * math.sqrt(T))
    vega_kappa = S * scalar_n_pdf(d1) * math.sqrt(T) * d1 / sigma
        
    return delta, gamma, theta, vega, rho, theoretical_price, delta_gamma, vega_kappa, 0.0

@njit
def calculate_greeks_array(S_arr, K_arr, T, r, sigma, is_call):
    n = len(S_arr)
    delta = np.empty(n)
    gamma = np.empty(n)
    theta = np.empty(n)
    vega = np.empty(n)
    rho = np.empty(n)
    price = np.empty(n)
    dg = np.empty(n)
    vk = np.empty(n)
    
    for i in range(n):
        d, g, t, v, rh, p, dga, vka, ts = calculate_greeks_scalar(S_arr[i], K_arr[i], T, r, sigma, is_call)
        delta[i] = d
        gamma[i] = g
        theta[i] = t
        vega[i] = v
        rho[i] = rh
        price[i] = p
        dg[i] = dga
        vk[i] = vka
        
    return delta, gamma, theta, vega, rho, price, dg, vk

@dataclass
class GreeksResult:
    """Greeks calculation result"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_vol: float
    theoretical_price: float
    delta_gamma: float
    vega_kappa: float
    timestamp_ns: int

class RealTimeGreeksCalculator:
    """Real-time options Greeks calculator with Numba JIT acceleration"""

    def __init__(self, use_cpp: bool = False):
        self.use_cpp = False # Using Numba for Elite Tier performance

    def calculate_greeks(self, S: Union[float, np.ndarray], K: Union[float, np.ndarray], 
                         T: float, r: float, sigma: float,
                         is_call: bool = True) -> Union[GreeksResult, Dict[str, np.ndarray]]:
        
        if isinstance(S, np.ndarray):
            d, g, t, v, rho, p, dg, vk = calculate_greeks_array(
                S, K, float(T), float(r), float(sigma), bool(is_call)
            )
            return {
                "delta": d, "gamma": g, "theta": t, "vega": v, "rho": rho,
                "theoretical_price": p, "delta_gamma": dg, "vega_kappa": vk
            }

        # Scalar path
        d, g, t, v, rho, p, dg, vk, ts = calculate_greeks_scalar(
            float(S), float(K), float(T), float(r), float(sigma), bool(is_call)
        )
        
        return GreeksResult(
            delta=d, gamma=g, theta=t, vega=v, rho=rho, 
            implied_vol=sigma, theoretical_price=p, 
            delta_gamma=dg, vega_kappa=vk, 
            timestamp_ns=int(datetime.now().timestamp()*1e9)
        )
