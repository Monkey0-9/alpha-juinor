"""
Pure Python Options Greeks Calculator - No C++ Dependency
Optimized with NumPy and vectorized operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

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

class PurePythonGreeksCalculator:
    """Pure Python options Greeks calculator with vectorized operations"""

    def __init__(self):
        """Initialize pure Python Greeks calculator"""
        self.cache = {}
        self.cache_size_limit = 10000

    def _norm_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized normal CDF using scipy or math

        Args:
            x: Input array

        Returns:
            Normal CDF values
        """
        try:
            from scipy.stats import norm
            return norm.cdf(x)
        except ImportError:
            import math
            if isinstance(x, np.ndarray):
                # Vectorized implementation
                return np.array([0.5 * (1 + math.erf(val / math.sqrt(2))) for val in x])
            else:
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorized normal PDF using numpy

        Args:
            x: Input array

        Returns:
            Normal PDF values
        """
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float,
                        is_call: bool = True) -> GreeksResult:
        """
        Calculate options Greeks using pure Python with NumPy optimization

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            is_call: True for call, False for put

        Returns:
            Greeks calculation result
        """
        # Check cache first
        cache_key = (S, K, T, r, sigma, is_call)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if T <= 0:
            result = GreeksResult(0, 0, 0, 0, 0, sigma, 0, 0, 0, 0)
            self._update_cache(cache_key, result)
            return result

        # Calculate d1 and d2
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Calculate CDF and PDF
        cnd_d1 = self._norm_cdf(np.array([d1]))[0]
        cnd_d2 = self._norm_cdf(np.array([d2]))[0]
        cnd_minus_d1 = self._norm_cdf(np.array([-d1]))[0]
        cnd_minus_d2 = self._norm_cdf(np.array([-d2]))[0]
        pdf_d1 = self._norm_pdf(np.array([d1]))[0]

        # Calculate Greeks
        if is_call:
            delta = cnd_d1
            theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * cnd_d2
            rho = K * T * np.exp(-r * T) * cnd_d2
        else:
            delta = cnd_d1 - 1.0
            theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * cnd_minus_d2
            rho = -K * T * np.exp(-r * T) * cnd_minus_d2

        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * pdf_d1 * sqrt_T / 100.0

        # Theoretical price
        if is_call:
            theoretical_price = S * cnd_d1 - K * np.exp(-r * T) * cnd_d2
        else:
            theoretical_price = K * np.exp(-r * T) * cnd_minus_d2 - S * cnd_minus_d1

        # Higher-order Greeks
        delta_gamma = -(d1 * pdf_d1) / (S**2 * sigma**2 * sqrt_T)
        vega_kappa = S * pdf_d1 * sqrt_T * d1 / sigma

        # Convert theta to per-day
        theta = theta / 365.0

        timestamp_ns = int(datetime.now().timestamp() * 1e9)

        result = GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            implied_vol=sigma,
            theoretical_price=theoretical_price,
            delta_gamma=delta_gamma,
            vega_kappa=vega_kappa,
            timestamp_ns=timestamp_ns
        )

        self._update_cache(cache_key, result)
        return result

    def calculate_batch_greeks(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Greeks for multiple options using vectorized operations

        Args:
            options_data: DataFrame with columns ['S', 'K', 'T', 'r', 'sigma', 'is_call']

        Returns:
            DataFrame with Greeks results
        """
        if len(options_data) == 0:
            return pd.DataFrame()

        # Extract arrays for vectorized calculation
        S = options_data['S'].values
        K = options_data['K'].values
        T = options_data['T'].values
        r = options_data['r'].values
        sigma = options_data['sigma'].values
        is_call = options_data['is_call'].values

        # Vectorized calculations
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Vectorized CDF and PDF
        cnd_d1 = self._norm_cdf(d1)
        cnd_d2 = self._norm_cdf(d2)
        cnd_minus_d1 = self._norm_cdf(-d1)
        cnd_minus_d2 = self._norm_cdf(-d2)
        pdf_d1 = self._norm_pdf(d1)

        # Vectorized Greeks calculation
        delta = np.where(is_call, cnd_d1, cnd_d1 - 1.0)
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * pdf_d1 * sqrt_T / 100.0

        # Vectorized theta calculation
        theta_call = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * T) * cnd_d2
        theta_put = -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * cnd_minus_d2
        theta = np.where(is_call, theta_call, theta_put) / 365.0

        # Vectorized rho calculation
        rho_call = K * T * np.exp(-r * T) * cnd_d2
        rho_put = -K * T * np.exp(-r * T) * cnd_minus_d2
        rho = np.where(is_call, rho_call, rho_put) / 100.0

        # Vectorized theoretical price
        theoretical_price = np.where(
            is_call,
            S * cnd_d1 - K * np.exp(-r * T) * cnd_d2,
            K * np.exp(-r * T) * cnd_minus_d2 - S * cnd_minus_d1
        )

        # Higher-order Greeks
        delta_gamma = -(d1 * pdf_d1) / (S**2 * sigma**2 * sqrt_T)
        vega_kappa = S * pdf_d1 * sqrt_T * d1 / sigma

        timestamp_ns = int(datetime.now().timestamp() * 1e9)

        # Create result DataFrame
        results = pd.DataFrame({
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'implied_vol': sigma,
            'theoretical_price': theoretical_price,
            'delta_gamma': delta_gamma,
            'vega_kappa': vega_kappa,
            'timestamp_ns': timestamp_ns
        })

        return results

    def calculate_portfolio_greeks(self, portfolio: List[Dict]) -> Dict:
        """
        Calculate portfolio-level Greeks using vectorized operations

        Args:
            portfolio: List of option positions with 'quantity', 'S', 'K', 'T', 'r', 'sigma', 'is_call'

        Returns:
            Portfolio Greeks
        """
        if not portfolio:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0,
                'theoretical_value': 0.0
            }

        # Convert to DataFrame for vectorized calculation
        df = pd.DataFrame(portfolio)

        # Calculate individual Greeks
        greeks_df = self.calculate_batch_greeks(df)

        # Apply quantities
        quantities = df['quantity'].values

        portfolio_greeks = {
            'delta': np.sum(greeks_df['delta'] * quantities),
            'gamma': np.sum(greeks_df['gamma'] * quantities),
            'theta': np.sum(greeks_df['theta'] * quantities),
            'vega': np.sum(greeks_df['vega'] * quantities),
            'rho': np.sum(greeks_df['rho'] * quantities),
            'theoretical_value': np.sum(greeks_df['theoretical_price'] * quantities)
        }

        return portfolio_greeks

    def calculate_implied_volatility(self, S: float, K: float, T: float, r: float,
                                    market_price: float, is_call: bool = True,
                                    max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiry
            r: Risk-free rate
            market_price: Market price of option
            is_call: True for call, False for put
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Implied volatility
        """
        # Initial guess
        sigma = 0.3

        for _ in range(max_iterations):
            greeks = self.calculate_greeks(S, K, T, r, sigma, is_call)

            # Calculate price difference
            price_diff = greeks.theoretical_price - market_price

            # Check convergence
            if abs(price_diff) < tolerance:
                return sigma

            # Newton-Raphson update
            if greeks.vega > 0:
                sigma -= price_diff / greeks.vega
            else:
                # Use bisection method if vega is too small
                break

        # Fallback to bisection method
        return self._bisection_iv(S, K, T, r, market_price, is_call)

    def _bisection_iv(self, S: float, K: float, T: float, r: float,
                      market_price: float, is_call: bool) -> float:
        """Bisection method for implied volatility"""
        low, high = 0.001, 5.0

        for _ in range(50):
            mid = (low + high) / 2
            greeks = self.calculate_greeks(S, K, T, r, mid, is_call)

            if greeks.theoretical_price > market_price:
                high = mid
            else:
                low = mid

            if high - low < 1e-6:
                break

        return (low + high) / 2

    def _update_cache(self, cache_key: tuple, result: GreeksResult):
        """Update cache with new result"""
        if len(self.cache) < self.cache_size_limit:
            self.cache[cache_key] = result

    def clear_cache(self):
        """Clear calculation cache"""
        self.cache.clear()
        logger.info("Pure Python Greeks calculation cache cleared")

# Performance benchmark function
def benchmark_greeks_calculator():
    """Benchmark the pure Python Greeks calculator"""
    import time

    calculator = PurePythonGreeksCalculator()

    # Benchmark single calculation
    start_time = time.time_ns()
    for _ in range(10000):
        calculator.calculate_greeks(100, 95, 0.25, 0.05, 0.2, True)
    end_time = time.time_ns()

    single_calc_time = (end_time - start_time) / 10000 / 1e6  # Convert to milliseconds

    # Benchmark batch calculation
    batch_data = pd.DataFrame({
        'S': np.random.uniform(90, 110, 1000),
        'K': np.random.uniform(90, 110, 1000),
        'T': np.random.uniform(0.1, 1.0, 1000),
        'r': np.random.uniform(0.01, 0.1, 1000),
        'sigma': np.random.uniform(0.1, 0.5, 1000),
        'is_call': np.random.choice([True, False], 1000),
        'quantity': np.random.randint(1, 100, 1000)
    })

    start_time = time.time_ns()
    calculator.calculate_batch_greeks(batch_data)
    end_time = time.time_ns()

    batch_calc_time = (end_time - start_time) / 1e6  # Convert to milliseconds

    return {
        'single_calc_avg_ms': single_calc_time,
        'batch_1000_calc_ms': batch_calc_time,
        'performance_ratio': batch_calc_time / (1000 * single_calc_time)
    }
