"""
Exotic Options Pricing
=======================

Pricing models for barrier, Asian, and lookback options using
Monte Carlo simulation and analytical approximations.

Features:
- Barrier options (knock-in, knock-out)
- Asian options (arithmetic and geometric average)
- Lookback options (fixed and floating strike)
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class BarrierOption:
    """Barrier option specification."""

    S: float  # Spot price
    K: float  # Strike price
    H: float  # Barrier level
    T: float  # Time to expiry (years)
    r: float  # Risk-free rate
    q: float  # Dividend yield
    sigma: float  # Volatility
    option_type: Literal["call", "put"]
    barrier_type: Literal["down-and-out", "down-and-in", "up-and-out", "up-and-in"]
    rebate: float = 0.0  # Rebate paid if barrier is hit


@dataclass
class AsianOption:
    """Asian option specification."""

    S: float  # Spot price
    K: float  # Strike price
    T: float  # Time to expiry (years)
    r: float  # Risk-free rate
    q: float  # Dividend yield
    sigma: float  # Volatility
    option_type: Literal["call", "put"]
    averaging_type: Literal["arithmetic", "geometric"] = "arithmetic"
    n_fixings: int = 252  # Number of fixings


@dataclass
class LookbackOption:
    """Lookback option specification."""

    S: float  # Spot price
    K: Optional[float]  # Strike price (None for floating strike)
    T: float  # Time to expiry (years)
    r: float  # Risk-free rate
    q: float  # Dividend yield
    sigma: float  # Volatility
    option_type: Literal["call", "put"]
    lookback_type: Literal["fixed", "floating"]


def price_barrier_option(option: BarrierOption, n_simulations: int = 100000) -> float:
    """
    Price barrier option using Monte Carlo simulation.

    Args:
        option: Barrier option specification
        n_simulations: Number of Monte Carlo paths

    Returns:
        Option price
    """
    dt = option.T / 252
    n_steps = int(option.T / dt)

    # Generate paths
    np.random.seed(42)
    Z = np.random.standard_normal((n_simulations, n_steps))

    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = option.S

    for t in range(1, n_steps + 1):
        drift = (option.r - option.q - 0.5 * option.sigma**2) * dt
        diffusion = option.sigma * np.sqrt(dt) * Z[:, t - 1]
        S[:, t] = S[:, t - 1] * np.exp(drift + diffusion)

    # Determine barrier hits
    if option.barrier_type.startswith("down"):
        barrier_hit = np.any(S <= option.H, axis=1)
    else:  # up barrier
        barrier_hit = np.any(S >= option.H, axis=1)

    # Calculate payoffs
    if option.option_type == "call":
        intrinsic = np.maximum(S[:, -1] - option.K, 0)
    else:
        intrinsic = np.maximum(option.K - S[:, -1], 0)

    # Apply barrier conditions
    if option.barrier_type.endswith("out"):
        # Knock-out: payoff only if barrier NOT hit
        payoff = np.where(barrier_hit, option.rebate, intrinsic)
    else:
        # Knock-in: payoff only if barrier IS hit
        payoff = np.where(barrier_hit, intrinsic, option.rebate)

    # Discount to present value
    price = np.exp(-option.r * option.T) * np.mean(payoff)

    return price


def price_asian_option(option: AsianOption, n_simulations: int = 100000) -> float:
    """
    Price Asian option using Monte Carlo simulation.

    Args:
        option: Asian option specification
        n_simulations: Number of Monte Carlo paths

    Returns:
        Option price
    """
    dt = option.T / option.n_fixings
    n_steps = option.n_fixings

    # Generate paths
    np.random.seed(42)
    Z = np.random.standard_normal((n_simulations, n_steps))

    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = option.S

    for t in range(1, n_steps + 1):
        drift = (option.r - option.q - 0.5 * option.sigma**2) * dt
        diffusion = option.sigma * np.sqrt(dt) * Z[:, t - 1]
        S[:, t] = S[:, t - 1] * np.exp(drift + diffusion)

    # Calculate average
    if option.averaging_type == "arithmetic":
        avg_price = np.mean(S[:, 1:], axis=1)
    else:  # geometric
        avg_price = np.exp(np.mean(np.log(S[:, 1:]), axis=1))

    # Calculate payoffs
    if option.option_type == "call":
        payoff = np.maximum(avg_price - option.K, 0)
    else:
        payoff = np.maximum(option.K - avg_price, 0)

    # Discount to present value
    price = np.exp(-option.r * option.T) * np.mean(payoff)

    return price


def price_lookback_option(option: LookbackOption, n_simulations: int = 100000) -> float:
    """
    Price lookback option using Monte Carlo simulation.

    Args:
        option: Lookback option specification
        n_simulations: Number of Monte Carlo paths

    Returns:
        Option price
    """
    dt = option.T / 252
    n_steps = int(option.T / dt)

    # Generate paths
    np.random.seed(42)
    Z = np.random.standard_normal((n_simulations, n_steps))

    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = option.S

    for t in range(1, n_steps + 1):
        drift = (option.r - option.q - 0.5 * option.sigma**2) * dt
        diffusion = option.sigma * np.sqrt(dt) * Z[:, t - 1]
        S[:, t] = S[:, t - 1] * np.exp(drift + diffusion)

    # Find extreme prices
    max_price = np.max(S, axis=1)
    min_price = np.min(S, axis=1)

    # Calculate payoffs
    if option.lookback_type == "fixed":
        if option.option_type == "call":
            payoff = np.maximum(max_price - option.K, 0)
        else:
            payoff = np.maximum(option.K - min_price, 0)
    else:  # floating strike
        if option.option_type == "call":
            payoff = S[:, -1] - min_price
        else:
            payoff = max_price - S[:, -1]

    # Discount to present value
    price = np.exp(-option.r * option.T) * np.mean(payoff)

    return price


def price_geometric_asian_closed_form(option: AsianOption) -> float:
    """
    Closed-form solution for geometric Asian option.

    More efficient than Monte Carlo for this specific case.

    Args:
        option: Asian option specification

    Returns:
        Option price
    """
    # Adjusted parameters for geometric average
    sigma_adj = option.sigma / np.sqrt(3)
    rho = 0.5 * (option.r - option.q - sigma_adj**2 / 6)

    d1 = (
        np.log(option.S / option.K)
        + (rho + 0.5 * sigma_adj**2) * option.T
    ) / (sigma_adj * np.sqrt(option.T))
    d2 = d1 - sigma_adj * np.sqrt(option.T)

    if option.option_type == "call":
        price = option.S * np.exp((rho - option.r) * option.T) * norm.cdf(
            d1
        ) - option.K * np.exp(-option.r * option.T) * norm.cdf(d2)
    else:
        price = option.K * np.exp(-option.r * option.T) * norm.cdf(
            -d2
        ) - option.S * np.exp((rho - option.r) * option.T) * norm.cdf(-d1)

    return price


class ExoticOptionsEngine:
    """
    Exotic options pricing engine with caching and analytics.
    """

    def __init__(self, n_simulations: int = 100000):
        self.n_simulations = n_simulations
        self._price_cache = {}

    def price(
        self, option: BarrierOption | AsianOption | LookbackOption
    ) -> dict:
        """
        Price exotic option with Greeks and analytics.

        Args:
            option: Option specification

        Returns:
            Dictionary with price and Greeks
        """
        # Calculate base price
        if isinstance(option, BarrierOption):
            price = price_barrier_option(option, self.n_simulations)
            option_type = "barrier"
        elif isinstance(option, AsianOption):
            if option.averaging_type == "geometric":
                price = price_geometric_asian_closed_form(option)
            else:
                price = price_asian_option(option, self.n_simulations)
            option_type = "asian"
        else:  # LookbackOption
            price = price_lookback_option(option, self.n_simulations)
            option_type = "lookback"

        # Calculate Greeks via finite differences
        greeks = self._calculate_greeks(option)

        return {
            "price": price,
            "type": option_type,
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "vega": greeks["vega"],
            "theta": greeks["theta"],
        }

    def _calculate_greeks(self, option) -> dict:
        """Calculate option Greeks using finite differences."""
        dS = option.S * 0.01  # 1% bump
        dsigma = 0.01  # 1% vol bump
        dt = 1 / 365  # 1 day

        # Delta (finite difference)
        option_up = type(option)(**{**vars(option), "S": option.S + dS})
        option_down = type(option)(**{**vars(option), "S": option.S - dS})

        if isinstance(option, BarrierOption):
            price_up = price_barrier_option(option_up, self.n_simulations // 2)
            price_down = price_barrier_option(option_down, self.n_simulations // 2)
            price_base = price_barrier_option(option, self.n_simulations // 2)
        elif isinstance(option, AsianOption):
            price_up = price_asian_option(option_up, self.n_simulations // 2)
            price_down = price_asian_option(option_down, self.n_simulations // 2)
            price_base = price_asian_option(option, self.n_simulations // 2)
        else:
            price_up = price_lookback_option(option_up, self.n_simulations // 2)
            price_down = price_lookback_option(option_down, self.n_simulations // 2)
            price_base = price_lookback_option(option, self.n_simulations // 2)

        delta = (price_up - price_down) / (2 * dS)
        gamma = (price_up - 2 * price_base + price_down) / (dS**2)

        # Vega
        option_vol_up = type(option)(
            **{**vars(option), "sigma": option.sigma + dsigma}
        )
        if isinstance(option, BarrierOption):
            price_vol_up = price_barrier_option(option_vol_up, self.n_simulations // 2)
        elif isinstance(option, AsianOption):
            price_vol_up = price_asian_option(option_vol_up, self.n_simulations // 2)
        else:
            price_vol_up = price_lookback_option(option_vol_up, self.n_simulations // 2)

        vega = (price_vol_up - price_base) / dsigma

        # Theta
        option_time_down = type(option)(**{**vars(option), "T": option.T - dt})
        if isinstance(option, BarrierOption):
            price_time_down = price_barrier_option(
                option_time_down, self.n_simulations // 2
            )
        elif isinstance(option, AsianOption):
            price_time_down = price_asian_option(
                option_time_down, self.n_simulations // 2
            )
        else:
            price_time_down = price_lookback_option(
                option_time_down, self.n_simulations // 2
            )

        theta = (price_time_down - price_base) / dt

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}
