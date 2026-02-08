"""
Volatility Surface Construction and Modeling
=============================================

Elite-tier implied volatility modeling using SABR and other
industry-standard models.

Models:
- SABR: Stochastic Alpha Beta Rho model for volatility smile
- Black-Scholes Implied Vol calculation
- Volatility surface interpolation

References:
- Hagan et al. (2002): "Managing Smile Risk"
- Gatheral (2006): "The Volatility Surface"
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata
from scipy.optimize import brentq, minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class OptionQuote:
    """Market quote for an option."""

    strike: float
    expiry_days: int
    option_type: str  # 'call' or 'put'
    price: float
    underlying_price: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0


class BlackScholesModel:
    """Black-Scholes option pricing and implied volatility."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return BlackScholesModel.d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def price(
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """
        Black-Scholes option pricing.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        d1 = BlackScholesModel.d1(S, K, T, r, q, sigma)
        d2 = BlackScholesModel.d2(S, K, T, r, q, sigma)

        if option_type.lower() == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(
                d2
            )
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(
                -q * T
            ) * norm.cdf(-d1)

        return price

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: str = "call",
    ) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method.

        Args:
            market_price: Observed market price
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            q: Dividend yield
            option_type: 'call' or 'put'

        Returns:
            Implied volatility or None if calculation fails
        """
        if T <= 0:
            return None

        try:

            def objective(sigma):
                return (
                    BlackScholesModel.price(S, K, T, r, q, sigma, option_type)
                    - market_price
                )

            # Brent's method requires bounds
            iv = brentq(objective, 0.001, 5.0)
            return iv
        except Exception as e:
            logger.warning(f"IV calculation failed: {e}")
            return None


class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) volatility model.

    Captures the volatility smile/skew commonly observed in
    options markets.

    Parameters:
        alpha: Initial volatility level
        beta: CEV parameter (0=normal, 1=lognormal)
        rho: Correlation between asset and volatility
        nu: Volatility of volatility
    """

    def __init__(self, alpha: float, beta: float, rho: float, nu: float):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_volatility(self, F: float, K: float, T: float) -> float:
        """
        Calculate implied volatility using SABR formula.

        Hagan approximation for implied volatility.

        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry (years)

        Returns:
            Implied volatility
        """
        alpha, beta, rho, nu = self.alpha, self.beta, self.rho, self.nu

        # Handle ATM case
        if abs(F - K) < 1e-6:
            FK_beta = F ** (1 - beta)
            term1 = alpha / FK_beta
            term2 = (
                1
                + (
                    ((1 - beta) ** 2 / 24) * (alpha**2 / FK_beta**2)
                    + (rho * beta * nu * alpha) / (4 * FK_beta)
                    + (2 - 3 * rho**2) * nu**2 / 24
                )
                * T
            )
            return term1 * term2

        # General case
        FK_mid = (F * K) ** ((1 - beta) / 2)
        log_FK = np.log(F / K)

        z = (nu / alpha) * FK_mid * log_FK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        numerator = alpha
        denominator = FK_mid * (
            1
            + ((1 - beta) ** 2 / 24) * log_FK**2
            + ((1 - beta) ** 4 / 1920) * log_FK**4
        )

        vol_atm = numerator / denominator
        vol_z = z / x_z if abs(x_z) > 1e-6 else 1.0

        sabr_correction = (
            1
            + (
                ((1 - beta) ** 2 / 24) * (alpha**2 / FK_mid**2)
                + (rho * beta * nu * alpha) / (4 * FK_mid)
                + (2 - 3 * rho**2) * nu**2 / 24
            )
            * T
        )

        return vol_atm * vol_z * sabr_correction

    @staticmethod
    def calibrate(
        F: float, strikes: np.ndarray, T: float, market_vols: np.ndarray, beta: float = 0.5
    ) -> "SABRModel":
        """
        Calibrate SABR parameters to market data.

        Args:
            F: Forward price
            strikes: Array of strike prices
            T: Time to expiry (years)
            market_vols: Market implied volatilities
            beta: Fixed beta parameter

        Returns:
            Calibrated SABRModel
        """

        def objective(params):
            alpha, rho, nu = params
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e10

            model = SABRModel(alpha, beta, rho, nu)
            model_vols = np.array([model.implied_volatility(F, K, T) for K in strikes])
            return np.sum((model_vols - market_vols) ** 2)

        # Initial guess
        atm_vol = market_vols[len(market_vols) // 2]
        initial_params = [atm_vol, 0.0, 0.3]

        result = minimize(
            objective,
            initial_params,
            method="L-BFGS-B",
            bounds=[(0.001, 2.0), (-0.999, 0.999), (0.001, 2.0)],
        )

        alpha, rho, nu = result.x
        return SABRModel(alpha, beta, rho, nu)


class VolatilitySurface:
    """
    Implied volatility surface for options pricing.

    Constructs a continuous surface from discrete market quotes.
    """

    def __init__(self):
        self.quotes: List[OptionQuote] = []
        self.surface_data: Optional[pd.DataFrame] = None
        self.interpolator = None

    def add_quote(self, quote: OptionQuote):
        """Add a market quote to the surface."""
        self.quotes.append(quote)

    def build_surface(self, method: str = "rbf"):
        """
        Build implied volatility surface from quotes.

        Args:
            method: Interpolation method ('rbf' or 'grid')
        """
        if not self.quotes:
            raise ValueError("No quotes available to build surface")

        # Calculate implied volatilities
        data = []
        for quote in self.quotes:
            T = quote.expiry_days / 365.0
            iv = BlackScholesModel.implied_volatility(
                quote.price,
                quote.underlying_price,
                quote.strike,
                T,
                quote.risk_free_rate,
                quote.dividend_yield,
                quote.option_type,
            )

            if iv is not None:
                moneyness = quote.strike / quote.underlying_price
                data.append(
                    {
                        "strike": quote.strike,
                        "expiry_days": quote.expiry_days,
                        "moneyness": moneyness,
                        "T": T,
                        "implied_vol": iv,
                    }
                )

        self.surface_data = pd.DataFrame(data)

        # Build interpolator
        if method == "rbf":
            points = self.surface_data[["moneyness", "T"]].values
            values = self.surface_data["implied_vol"].values
            self.interpolator = RBFInterpolator(points, values, smoothing=0.1)
        else:
            # Grid interpolation fallback
            self.interpolator = None

        logger.info(f"Built volatility surface with {len(data)} points")

    def get_vol(self, strike: float, expiry_days: int, underlying_price: float) -> float:
        """
        Get implied volatility from the surface.

        Args:
            strike: Strike price
            expiry_days: Days to expiry
            underlying_price: Current underlying price

        Returns:
            Interpolated implied volatility
        """
        if self.interpolator is None:
            raise ValueError("Surface not built. Call build_surface() first.")

        moneyness = strike / underlying_price
        T = expiry_days / 365.0

        try:
            vol = self.interpolator([[moneyness, T]])[0]
            return max(0.01, vol)  # Floor at 1%
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}")
            return 0.3  # Default vol

    def get_vol_smile(self, expiry_days: int, underlying_price: float) -> pd.DataFrame:
        """
        Get volatility smile for a specific expiry.

        Args:
            expiry_days: Days to expiry
            underlying_price: Current underlying price

        Returns:
            DataFrame with strikes and implied vols
        """
        strikes = np.linspace(
            underlying_price * 0.8, underlying_price * 1.2, 20
        )
        vols = [self.get_vol(K, expiry_days, underlying_price) for K in strikes]

        return pd.DataFrame({"strike": strikes, "implied_vol": vols})

    def get_term_structure(self, strike: float, underlying_price: float) -> pd.DataFrame:
        """
        Get volatility term structure for a specific strike.

        Args:
            strike: Strike price
            underlying_price: Current underlying price

        Returns:
            DataFrame with expiries and implied vols
        """
        expiries = np.arange(1, 365, 7)  # Weekly expiries
        vols = [self.get_vol(strike, T, underlying_price) for T in expiries]

        return pd.DataFrame({"expiry_days": expiries, "implied_vol": vols})
