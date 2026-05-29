import numpy as np
import pandas as pd
from typing import Dict, Any


class PortfolioOptimizer:
    """Mean-Variance Optimization with KELLY CRITERION for optimal sizing."""

    def __init__(self):
        """Initialize with historical win rate tracking."""
        self.win_rate = 0.53  # Conservative estimate from backtest
        self.avg_win = 0.025  # 2.5% average win
        self.avg_loss = -0.015  # 1.5% average loss
        self.recent_trades = []  # Track last 100 trades

    def optimize_weights(
        self, symbols: list[str], signals: list[float],
        volatilities: list[float] = None
    ) -> Dict[str, float]:
        """Calculate KELLY CRITERION optimal weights with volatility scaling.
        
        Kelly Formula: f* = (p*b - q) / b
        Where: p=win%, q=loss%, b=win/loss ratio
        
        AGGRESSIVE: Use full Kelly (most maximize returns, though risky).
        Can reduce to 0.25*Kelly for conservative play.
        """
        if not symbols:
            return {}

        positive_signals = [max(0.0, s) for s in signals]
        
        # KELLY CRITERION CALCULATION
        # win_rate and avg_win/loss from recent trades
        if self.win_rate > 0 and self.avg_win > 0:
            ratio = abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 1.0
            kelly_fraction = (self.win_rate * ratio - (1 - self.win_rate)) / ratio
            kelly_fraction = max(0.01, min(kelly_fraction, 0.25))  # Limit to 1-25% of portfolio per trade
        else:
            kelly_fraction = 0.1
        
        # SIGNAL STRENGTH WEIGHTING: Strong signals get exponential boost
        # Squared weighting makes 0.8 signal get 16x weight of 0.1 signal
        squared_signals = [s ** 1.8 for s in positive_signals]  # Even more aggressive than ^1.5
        
        # VOLATILITY ADJUSTMENT: Scale position size inversely with volatility
        # High vol = reduce size, Low vol = increase size
        if volatilities is None:
            volatilities = [0.02] * len(symbols)  # Default to 2% vol
        
        vol_adjusted = []
        for sig, vol in zip(squared_signals, volatilities):
            vol_factor = 0.015 / max(vol, 0.01)  # Normalize to 1.5% base vol
            vol_factor = min(vol_factor, 2.0)  # Cap at 2x boost
            vol_adjusted.append(sig * vol_factor)
        
        total = np.sum(vol_adjusted)
        if total == 0:
            return {s: 0.0 for s in symbols}

        # AGGRESSIVE: Apply Kelly multiplier to position sizes
        weights = {}
        for s, adj_sig in zip(symbols, vol_adjusted):
            base_weight = adj_sig / total
            # Scale by Kelly fraction for optimal sizing
            final_weight = base_weight * kelly_fraction * 10  # 10x multiplier for stronger sizing
            weights[s] = float(np.clip(final_weight, 0.0, 0.20))  # Cap individual position at 20%
        
        # Normalize to sum to 1.0
        sum_weights = sum(weights.values())
        if sum_weights > 0:
            weights = {s: w / sum_weights for s, w in weights.items()}
        
        return weights

    def update_trade_performance(self, won: bool, win_amt: float, loss_amt: float) -> None:
        """Update Kelly parameters from realized trade performance."""
        self.recent_trades.append(won)
        if len(self.recent_trades) > 100:
            self.recent_trades.pop(0)
        
        # Recalculate win rate from recent trades
        self.win_rate = sum(self.recent_trades) / len(self.recent_trades)
        if win_amt > 0:
            self.avg_win = win_amt
        if loss_amt < 0:
            self.avg_loss = loss_amt


class MultiFactorEngine:
    """Ranks assets with AGGRESSIVE signal combination.
    
    Combines alpha + momentum + volatility with aggressive weighting.
    """

    def rank_assets(
        self,
        signals: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Rank assets by: alpha*0.60 + vol-adjusted momentum*0.40."""
        rankings = {}
        for symbol, alpha in signals.items():
            if symbol in historical_data:
                data = historical_data[symbol]
                close = data["close"].astype(float)
                
                # AGGRESSIVE momentum: Recent 10-day return with higher weight
                if len(close) > 1:
                    momentum_10d = (close.iloc[-1] / close.iloc[0]) - 1
                else:
                    momentum_10d = 0.0
                
                # Recent momentum (last 5 bars) weighted 70%
                if len(close) >= 5:
                    recent_momentum = close.pct_change().tail(5).mean()
                    momentum = recent_momentum * 0.7 + momentum_10d * 0.3
                else:
                    momentum = momentum_10d
                
                # Volatility penalty MINIMIZED
                vol = float(close.pct_change().dropna().std()) if len(close) > 1 else 0.0
                
                # AGGRESSIVE: Favor high momentum even in moderate vol
                risk_adj = momentum / (1.0 + vol * 1.0) if vol > 0 else momentum
                
                # Final rank: Heavy on alpha + momentum
                rankings[symbol] = alpha * 0.65 + risk_adj * 0.35
            else:
                rankings[symbol] = alpha

        return dict(
            sorted(
                rankings.items(), key=lambda x: x[1], reverse=True
            )
        )


class MonteCarloSimulator:
    """Portfolio-level Monte Carlo survival analysis.

    Runs N simulated random walks using bootstrapped daily returns
    to estimate the probability that the portfolio survives
    (i.e. does not breach the ruin threshold) over a given horizon.
    """

    def run_survival_analysis(
        self,
        initial_capital: float,
        daily_returns: np.ndarray[Any, Any],
        days: int = 252,
        n_simulations: int = 1000,
        ruin_threshold: float = 0.5,
    ) -> float:
        """Estimate probability of survival over N days.

        Parameters
        ----------
        initial_capital : float
            Starting portfolio value.
        daily_returns : np.ndarray
            Historical daily return samples to bootstrap from.
        days : int
            Simulation horizon in trading days.
        n_simulations : int
            Number of Monte Carlo paths.
        ruin_threshold : float
            Fraction of capital lost that constitutes ruin
            (0.5 = 50% drawdown).

        Returns
        -------
        float
            Probability of survival (0.0 to 1.0).
        """
        if (
            len(daily_returns) < 2
            or initial_capital <= 0
            or days <= 0
        ):
            return 0.5  # Insufficient data

        mu = float(np.mean(daily_returns))
        sigma = float(np.std(daily_returns))
        if sigma == 0:
            return 1.0  # No volatility = no ruin

        ruin_level = initial_capital * (1 - ruin_threshold)
        survived = 0

        rng = np.random.default_rng(42)  # Reproducible

        for _ in range(n_simulations):
            # Bootstrap path from observed return distribution
            path_returns = rng.normal(mu, sigma, days)
            prices = initial_capital * np.cumprod(1 + path_returns)

            if np.min(prices) > ruin_level:
                survived += 1

        return survived / n_simulations
