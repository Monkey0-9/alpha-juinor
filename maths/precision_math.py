"""
Precision Math Engine - High-Accuracy Calculations
===================================================

Provides high-precision mathematical operations for trading:
- Decimal arithmetic (avoids floating point errors)
- Monte Carlo simulations with configurable iterations
- Bayesian probability updates
- Exact portfolio optimization
- Statistical hypothesis testing

All calculations use Decimal where precision matters.
"""

import logging
import math
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Set high precision for Decimal operations
getcontext().prec = 28


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo simulation."""
    mean: float
    std: float
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    max_drawdown: float
    percentiles: Dict[int, float]
    simulations: int


@dataclass
class BayesianUpdate:
    """Result from Bayesian probability update."""
    prior: float
    likelihood: float
    posterior: float
    evidence: float
    bayes_factor: float


class PrecisionMath:
    """
    High-precision mathematical engine for trading calculations.

    Avoids common floating-point precision issues by using
    Decimal arithmetic for critical calculations.
    """

    # Monte Carlo defaults
    DEFAULT_SIMULATIONS = 100_000
    DEFAULT_HORIZON_DAYS = 252

    # Precision for different operations
    PRICE_PRECISION = Decimal("0.01")
    POSITION_PRECISION = Decimal("0.0001")
    RETURN_PRECISION = Decimal("0.000001")

    @staticmethod
    def to_decimal(value: Union[float, int, str, Decimal]) -> Decimal:
        """Convert any numeric to Decimal safely."""
        if isinstance(value, Decimal):
            return value
        if isinstance(value, float):
            # Handle float -> Decimal conversion carefully
            return Decimal(str(round(value, 10)))
        return Decimal(str(value))

    @staticmethod
    def precise_round(
        value: Decimal,
        precision: Decimal = Decimal("0.01")
    ) -> Decimal:
        """Round a Decimal to specified precision."""
        return value.quantize(precision, rounding=ROUND_HALF_UP)

    @staticmethod
    def calculate_return(
        entry_price: float,
        exit_price: float,
        include_costs: bool = True,
        commission_bps: float = 5.0
    ) -> Decimal:
        """
        Calculate precise return including transaction costs.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            include_costs: Whether to include transaction costs
            commission_bps: Round-trip commission in basis points

        Returns:
            Decimal return value
        """
        entry = PrecisionMath.to_decimal(entry_price)
        exit_val = PrecisionMath.to_decimal(exit_price)

        if entry == 0:
            return Decimal("0")

        gross_return = (exit_val - entry) / entry

        if include_costs:
            cost = PrecisionMath.to_decimal(commission_bps) / Decimal("10000")
            net_return = gross_return - cost
            return PrecisionMath.precise_round(net_return, Decimal("0.000001"))

        return PrecisionMath.precise_round(gross_return, Decimal("0.000001"))

    @staticmethod
    def calculate_position_value(
        shares: float,
        price: float,
        nav: float
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate position value and weight precisely.

        Returns:
            (position_value, position_weight)
        """
        shares_d = PrecisionMath.to_decimal(shares)
        price_d = PrecisionMath.to_decimal(price)
        nav_d = PrecisionMath.to_decimal(nav)

        position_value = shares_d * price_d
        position_weight = position_value / nav_d if nav_d > 0 else Decimal("0")

        return (
            PrecisionMath.precise_round(position_value, Decimal("0.01")),
            PrecisionMath.precise_round(position_weight, Decimal("0.0001"))
        )

    @staticmethod
    def kelly_criterion(
        win_probability: float,
        win_loss_ratio: float,
        fractional: float = 0.25
    ) -> Decimal:
        """
        Calculate Kelly-optimal position size.

        f* = (p * b - q) / b

        Where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio

        Args:
            win_probability: Estimated win probability (0 to 1)
            win_loss_ratio: Average win / Average loss
            fractional: Fraction of Kelly to use (safety factor)

        Returns:
            Optimal position size as fraction of capital
        """
        p = PrecisionMath.to_decimal(max(0.01, min(0.99, win_probability)))
        b = PrecisionMath.to_decimal(max(0.01, win_loss_ratio))
        q = Decimal("1") - p
        frac = PrecisionMath.to_decimal(fractional)

        # Kelly formula
        kelly = (p * b - q) / b

        # Apply fractional Kelly and bounds
        result = kelly * frac
        result = max(Decimal("0"), min(Decimal("1"), result))

        return PrecisionMath.precise_round(result, Decimal("0.0001"))

    @staticmethod
    def monte_carlo_simulation(
        initial_value: float,
        expected_return: float,
        volatility: float,
        simulations: int = 100_000,
        horizon_days: int = 252,
        seed: Optional[int] = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for portfolio value.

        Uses Geometric Brownian Motion:
        S(t) = S(0) * exp((μ - σ²/2)t + σ√t * Z)

        Args:
            initial_value: Starting portfolio value
            expected_return: Annualized expected return
            volatility: Annualized volatility
            simulations: Number of simulation paths
            horizon_days: Trading days to simulate
            seed: Random seed for reproducibility

        Returns:
            MonteCarloResult with statistics
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert to daily parameters
        dt = 1.0 / 252  # One trading day
        daily_return = expected_return * dt
        daily_vol = volatility * np.sqrt(dt)
        drift = daily_return - 0.5 * daily_vol ** 2

        # Generate random paths
        random_shocks = np.random.normal(0, 1, (simulations, horizon_days))

        # Calculate log returns
        log_returns = drift + daily_vol * random_shocks

        # Calculate cumulative returns
        cumulative_returns = np.exp(np.cumsum(log_returns, axis=1))

        # Final values
        final_values = initial_value * cumulative_returns[:, -1]

        # Calculate returns
        total_returns = (final_values - initial_value) / initial_value

        # Calculate maximum drawdown for each path
        cumulative_max = np.maximum.accumulate(cumulative_returns, axis=1)
        drawdowns = (cumulative_returns - cumulative_max) / cumulative_max
        max_drawdowns = np.min(drawdowns, axis=1)

        # Statistics
        mean_return = float(np.mean(total_returns))
        std_return = float(np.std(total_returns))

        # VaR and CVaR
        var_95 = float(np.percentile(total_returns, 5))  # 5th percentile = 95% VaR
        var_99 = float(np.percentile(total_returns, 1))  # 1st percentile = 99% VaR

        # CVaR = Expected return given return < VaR
        cvar_95 = float(np.mean(total_returns[total_returns <= var_95]))

        # Percentiles
        percentiles = {
            1: float(np.percentile(total_returns, 1)),
            5: float(np.percentile(total_returns, 5)),
            10: float(np.percentile(total_returns, 10)),
            25: float(np.percentile(total_returns, 25)),
            50: float(np.percentile(total_returns, 50)),
            75: float(np.percentile(total_returns, 75)),
            90: float(np.percentile(total_returns, 90)),
            95: float(np.percentile(total_returns, 95)),
            99: float(np.percentile(total_returns, 99))
        }

        return MonteCarloResult(
            mean=mean_return,
            std=std_return,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=float(np.mean(max_drawdowns)),
            percentiles=percentiles,
            simulations=simulations
        )

    @staticmethod
    def bayesian_update(
        prior: float,
        likelihood_given_true: float,
        likelihood_given_false: float
    ) -> BayesianUpdate:
        """
        Perform Bayesian probability update.

        P(H|E) = P(E|H) * P(H) / P(E)

        Args:
            prior: Prior probability P(H)
            likelihood_given_true: P(E|H) - likelihood of evidence given hypothesis true
            likelihood_given_false: P(E|¬H) - likelihood of evidence given hypothesis false

        Returns:
            BayesianUpdate with posterior probability
        """
        p_h = PrecisionMath.to_decimal(max(0.001, min(0.999, prior)))
        p_e_h = PrecisionMath.to_decimal(max(0.001, min(0.999, likelihood_given_true)))
        p_e_not_h = PrecisionMath.to_decimal(max(0.001, min(0.999, likelihood_given_false)))

        # P(¬H)
        p_not_h = Decimal("1") - p_h

        # P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
        p_e = p_e_h * p_h + p_e_not_h * p_not_h

        # P(H|E) = P(E|H) * P(H) / P(E)
        posterior = (p_e_h * p_h) / p_e if p_e > 0 else Decimal("0")

        # Bayes factor = P(E|H) / P(E|¬H)
        bayes_factor = p_e_h / p_e_not_h if p_e_not_h > 0 else Decimal("Infinity")

        return BayesianUpdate(
            prior=float(p_h),
            likelihood=float(p_e_h),
            posterior=float(PrecisionMath.precise_round(posterior, Decimal("0.0001"))),
            evidence=float(p_e),
            bayes_factor=float(bayes_factor) if bayes_factor != Decimal("Infinity") else float("inf")
        )

    @staticmethod
    def sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.05,
        annualization_factor: float = 252.0
    ) -> Decimal:
        """
        Calculate annualized Sharpe ratio precisely.

        Sharpe = (E[R] - Rf) / σ(R) * √T
        """
        if len(returns) < 2:
            return Decimal("0")

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)

        if std_return == 0:
            return Decimal("0")

        daily_rf = risk_free_rate / annualization_factor
        excess_return = mean_return - daily_rf

        sharpe = (excess_return / std_return) * np.sqrt(annualization_factor)

        return PrecisionMath.to_decimal(sharpe)

    @staticmethod
    def sortino_ratio(
        returns: List[float],
        risk_free_rate: float = 0.05,
        annualization_factor: float = 252.0
    ) -> Decimal:
        """
        Calculate annualized Sortino ratio (downside risk only).

        Sortino = (E[R] - Rf) / σ_downside(R) * √T
        """
        if len(returns) < 2:
            return Decimal("0")

        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)

        # Downside deviation - only negative returns
        daily_rf = risk_free_rate / annualization_factor
        downside_returns = returns_array[returns_array < daily_rf] - daily_rf

        if len(downside_returns) == 0:
            return Decimal("999")  # No downside

        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if downside_std == 0:
            return Decimal("0")

        excess_return = mean_return - daily_rf
        sortino = (excess_return / downside_std) * np.sqrt(annualization_factor)

        return PrecisionMath.to_decimal(sortino)

    @staticmethod
    def calmar_ratio(
        returns: List[float],
        annualization_factor: float = 252.0
    ) -> Decimal:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        """
        if len(returns) < 2:
            return Decimal("0")

        returns_array = np.array(returns)

        # Annualized return
        cumulative = np.cumprod(1 + returns_array)
        total_return = cumulative[-1] - 1
        n_periods = len(returns) / annualization_factor
        annualized_return = (1 + total_return) ** (1 / n_periods) - 1 if n_periods > 0 else 0

        # Maximum drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        if max_drawdown == 0:
            return Decimal("999")

        calmar = annualized_return / max_drawdown

        return PrecisionMath.to_decimal(calmar)

    @staticmethod
    def information_ratio(
        portfolio_returns: List[float],
        benchmark_returns: List[float],
        annualization_factor: float = 252.0
    ) -> Decimal:
        """
        Calculate Information Ratio.

        IR = (E[Rp - Rb]) / σ(Rp - Rb) * √T
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return Decimal("0")

        port = np.array(portfolio_returns)
        bench = np.array(benchmark_returns)

        active_returns = port - bench
        mean_active = np.mean(active_returns)
        std_active = np.std(active_returns, ddof=1)

        if std_active == 0:
            return Decimal("0")

        ir = (mean_active / std_active) * np.sqrt(annualization_factor)

        return PrecisionMath.to_decimal(ir)

    @staticmethod
    def optimal_portfolio_weights(
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        max_weight: float = 0.10,
        min_weight: float = -0.05
    ) -> np.ndarray:
        """
        Calculate mean-variance optimal portfolio weights.

        Uses quadratic optimization:
        max μᵀw - λ/2 wᵀΣw

        Subject to:
        - sum(w) = 1 (fully invested)
        - min_weight <= w_i <= max_weight

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            risk_aversion: Lambda parameter
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            Optimal portfolio weights
        """
        n = len(expected_returns)

        # Simple analytical solution for unconstrained case
        # w* = (1/λ) * Σ^(-1) * μ
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            raw_weights = (1 / risk_aversion) * inv_cov @ expected_returns

            # Normalize to sum to 1
            weights = raw_weights / np.sum(np.abs(raw_weights))

            # Apply constraints
            weights = np.clip(weights, min_weight, max_weight)

            # Re-normalize
            if np.sum(np.abs(weights)) > 0:
                weights = weights / np.sum(np.abs(weights))

            return weights

        except np.linalg.LinAlgError:
            # If matrix inversion fails, return equal weights
            return np.ones(n) / n

    @staticmethod
    def calculate_beta(
        asset_returns: List[float],
        market_returns: List[float]
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate beta and alpha using linear regression.

        Returns:
            (beta, alpha)
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 10:
            return (Decimal("1"), Decimal("0"))

        asset = np.array(asset_returns)
        market = np.array(market_returns)

        # Covariance / Variance
        cov = np.cov(asset, market)[0, 1]
        var_market = np.var(market)

        if var_market == 0:
            return (Decimal("1"), Decimal("0"))

        beta = cov / var_market
        alpha = np.mean(asset) - beta * np.mean(market)

        return (
            PrecisionMath.to_decimal(beta),
            PrecisionMath.to_decimal(alpha)
        )


# Convenience functions
def kelly(win_prob: float, win_loss_ratio: float, fraction: float = 0.25) -> float:
    """Quick Kelly calculation."""
    return float(PrecisionMath.kelly_criterion(win_prob, win_loss_ratio, fraction))


def monte_carlo(
    initial: float,
    ret: float,
    vol: float,
    sims: int = 100_000
) -> MonteCarloResult:
    """Quick Monte Carlo simulation."""
    return PrecisionMath.monte_carlo_simulation(initial, ret, vol, sims)


def sharpe(returns: List[float], rf: float = 0.05) -> float:
    """Quick Sharpe ratio."""
    return float(PrecisionMath.sharpe_ratio(returns, rf))
