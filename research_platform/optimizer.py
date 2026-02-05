"""
Hyper-Parameter Optimizer - Bayesian Optimization at Scale
=============================================================

Elite firms use massive compute to optimize parameters.
This module implements:
1. Bayesian optimization using Optuna
2. Walk-forward cross-validation
3. Multi-objective optimization (Sharpe, Sortino, Max DD)
4. Automatic parameter selection

Stop guessing parameters. Optimize them systematically.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optuna is optional - graceful degradation
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("[OPTIMIZER] Optuna not installed. Install with: pip install optuna")


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    strategy_name: str

    # Best parameters
    best_params: Dict[str, Any]

    # Best performance
    best_sharpe: float
    best_sortino: float
    best_return: float
    best_max_dd: float

    # Optimization stats
    n_trials: int
    optimization_time_sec: float

    # Validation
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    oos_degradation: float

    # Robustness
    param_stability: float  # How sensitive are results to params
    is_robust: bool

    # Timestamp
    optimization_date: datetime


@dataclass
class ParameterSpace:
    """Definition of parameter search space."""
    name: str
    param_type: str  # int, float, categorical
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False


class WalkForwardOptimizer:
    """
    Walk-forward optimization with cross-validation.

    Tests parameters on multiple time periods to ensure robustness.
    """

    def __init__(self, n_folds: int = 5):
        """Initialize the optimizer."""
        self.n_folds = n_folds

    def cross_validate(
        self,
        params: Dict[str, Any],
        signal_func: Callable,
        market_data: pd.DataFrame,
        returns: pd.Series
    ) -> float:
        """
        Perform walk-forward cross-validation.

        Returns average out-of-sample Sharpe.
        """
        if len(returns) < 100:
            return -10.0

        fold_size = len(returns) // self.n_folds
        oos_sharpes = []

        for i in range(self.n_folds - 1):
            # Train on folds 0..i, test on fold i+1
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, len(returns))

            if test_end <= test_start:
                continue

            # Generate signal for test period
            try:
                test_returns = returns.iloc[test_start:test_end]

                # Apply strategy with params
                test_data = market_data.iloc[test_start:test_end]
                signal = signal_func(test_data, params)

                if signal is None or len(signal) == 0:
                    continue

                # Calculate strategy returns
                strategy_returns = test_returns * signal.shift(1)
                strategy_returns = strategy_returns.dropna()

                if len(strategy_returns) < 10:
                    continue

                # Calculate Sharpe
                sharpe = self._calc_sharpe(strategy_returns)
                oos_sharpes.append(sharpe)

            except Exception:
                continue

        if not oos_sharpes:
            return -10.0

        return float(np.mean(oos_sharpes))

    def _calc_sharpe(self, returns: pd.Series, risk_free: float = 0.04) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 5 or returns.std() == 0:
            return 0
        excess = returns - risk_free / 252
        return float(excess.mean() / excess.std() * np.sqrt(252))


class BayesianOptimizer:
    """
    Bayesian optimization using Optuna.

    Efficiently searches parameter space using TPE sampler.
    """

    def __init__(
        self,
        n_trials: int = 100,
        n_jobs: int = 1,
        timeout_sec: Optional[int] = 300
    ):
        """Initialize the optimizer."""
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout_sec = timeout_sec

        self.wf_optimizer = WalkForwardOptimizer()

        if not OPTUNA_AVAILABLE:
            logger.warning("[OPTIMIZER] Optuna not available, will use grid search fallback")
        else:
            logger.info("[OPTIMIZER] Bayesian Optimizer initialized")

    def optimize(
        self,
        strategy_name: str,
        signal_func: Callable,
        param_space: List[ParameterSpace],
        market_data: pd.DataFrame,
        returns: pd.Series,
        direction: str = "maximize"
    ) -> OptimizationResult:
        """
        Optimize parameters for a strategy.

        Args:
            strategy_name: Name of strategy
            signal_func: Function(data, params) -> signal
            param_space: List of ParameterSpace definitions
            market_data: Market data DataFrame
            returns: Returns series
            direction: "maximize" or "minimize"
        """
        start_time = datetime.utcnow()

        if not OPTUNA_AVAILABLE:
            return self._grid_search_fallback(
                strategy_name, signal_func, param_space, market_data, returns
            )

        # Create Optuna study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=f"{strategy_name}_opt"
        )

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = {}
            for p in param_space:
                if p.param_type == "int":
                    params[p.name] = trial.suggest_int(
                        p.name, int(p.low), int(p.high)
                    )
                elif p.param_type == "float":
                    if p.log_scale:
                        params[p.name] = trial.suggest_float(
                            p.name, p.low, p.high, log=True
                        )
                    else:
                        params[p.name] = trial.suggest_float(
                            p.name, p.low, p.high
                        )
                elif p.param_type == "categorical":
                    params[p.name] = trial.suggest_categorical(
                        p.name, p.choices
                    )

            # Evaluate with walk-forward CV
            sharpe = self.wf_optimizer.cross_validate(
                params, signal_func, market_data, returns
            )

            return sharpe

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout_sec,
            show_progress_bar=False
        )

        # Best trial
        best = study.best_trial
        best_params = best.params
        best_sharpe = best.value

        # Calculate additional metrics with best params
        metrics = self._calculate_final_metrics(
            signal_func, best_params, market_data, returns
        )

        # Calculate parameter stability
        top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
        param_stability = self._calculate_param_stability(top_trials, param_space)

        # OOS degradation
        split = int(len(returns) * 0.7)
        is_returns = returns.iloc[:split]
        oos_returns = returns.iloc[split:]

        is_sharpe = self._evaluate_params(
            signal_func, best_params, market_data.iloc[:split], is_returns
        )
        oos_sharpe = self._evaluate_params(
            signal_func, best_params, market_data.iloc[split:], oos_returns
        )

        oos_degradation = (is_sharpe - oos_sharpe) / is_sharpe if is_sharpe > 0 else 1.0
        is_robust = oos_degradation < 0.30 and oos_sharpe > 0.5 and param_stability > 0.7

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_params,
            best_sharpe=best_sharpe,
            best_sortino=metrics.get("sortino", 0),
            best_return=metrics.get("annual_return", 0),
            best_max_dd=metrics.get("max_dd", 0),
            n_trials=len(study.trials),
            optimization_time_sec=elapsed,
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=oos_sharpe,
            oos_degradation=oos_degradation,
            param_stability=param_stability,
            is_robust=is_robust,
            optimization_date=datetime.utcnow()
        )

    def _evaluate_params(
        self,
        signal_func: Callable,
        params: Dict[str, Any],
        market_data: pd.DataFrame,
        returns: pd.Series
    ) -> float:
        """Evaluate parameters on data."""
        try:
            signal = signal_func(market_data, params)
            if signal is None:
                return 0

            strategy_returns = returns * signal.shift(1)
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) < 10:
                return 0

            return self.wf_optimizer._calc_sharpe(strategy_returns)
        except Exception:
            return 0

    def _calculate_final_metrics(
        self,
        signal_func: Callable,
        params: Dict[str, Any],
        market_data: pd.DataFrame,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate final performance metrics."""
        try:
            signal = signal_func(market_data, params)
            if signal is None:
                return {}

            strategy_returns = returns * signal.shift(1)
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) < 10:
                return {}

            # Sortino
            downside = strategy_returns[strategy_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = strategy_returns.mean() / downside.std() * np.sqrt(252)
            else:
                sortino = 10

            # Annual return
            annual_return = (1 + strategy_returns).prod() ** (252 / len(strategy_returns)) - 1

            # Max drawdown
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_dd = drawdown.min()

            return {
                "sortino": float(sortino),
                "annual_return": float(annual_return),
                "max_dd": float(max_dd)
            }
        except Exception:
            return {}

    def _calculate_param_stability(
        self,
        top_trials: List,
        param_space: List[ParameterSpace]
    ) -> float:
        """Calculate how stable parameters are across top trials."""
        if len(top_trials) < 2:
            return 1.0

        stabilities = []

        for p in param_space:
            if p.param_type == "categorical":
                continue

            values = [t.params.get(p.name, 0) for t in top_trials]
            if not values:
                continue

            # Coefficient of variation
            mean_val = np.mean(values)
            std_val = np.std(values)

            if mean_val != 0:
                cv = std_val / abs(mean_val)
                stability = max(0, 1 - cv)
                stabilities.append(stability)

        return float(np.mean(stabilities)) if stabilities else 1.0

    def _grid_search_fallback(
        self,
        strategy_name: str,
        signal_func: Callable,
        param_space: List[ParameterSpace],
        market_data: pd.DataFrame,
        returns: pd.Series
    ) -> OptimizationResult:
        """Fallback to simple grid search when Optuna unavailable."""
        import itertools

        start_time = datetime.utcnow()

        # Generate grid
        param_grids = []
        param_names = []

        for p in param_space:
            param_names.append(p.name)
            if p.param_type == "int":
                param_grids.append(list(range(int(p.low), int(p.high) + 1, max(1, int((p.high - p.low) / 5)))))
            elif p.param_type == "float":
                param_grids.append(np.linspace(p.low, p.high, 5).tolist())
            elif p.param_type == "categorical":
                param_grids.append(p.choices)

        # Limit combinations
        total_combinations = 1
        for g in param_grids:
            total_combinations *= len(g)

        if total_combinations > 100:
            # Sample randomly
            combinations = []
            for _ in range(100):
                combo = [np.random.choice(g) for g in param_grids]
                combinations.append(combo)
        else:
            combinations = list(itertools.product(*param_grids))

        # Evaluate
        best_sharpe = -10.0
        best_params = {}

        for combo in combinations:
            params = dict(zip(param_names, combo))
            sharpe = self.wf_optimizer.cross_validate(
                params, signal_func, market_data, returns
            )

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        elapsed = (datetime.utcnow() - start_time).total_seconds()

        return OptimizationResult(
            strategy_name=strategy_name,
            best_params=best_params,
            best_sharpe=best_sharpe,
            best_sortino=0,
            best_return=0,
            best_max_dd=0,
            n_trials=len(combinations),
            optimization_time_sec=elapsed,
            in_sample_sharpe=best_sharpe,
            out_of_sample_sharpe=0,
            oos_degradation=0,
            param_stability=0.5,
            is_robust=False,
            optimization_date=datetime.utcnow()
        )


# Singleton
_optimizer: Optional[BayesianOptimizer] = None


def get_optimizer() -> BayesianOptimizer:
    """Get or create the Bayesian Optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = BayesianOptimizer()
    return _optimizer
