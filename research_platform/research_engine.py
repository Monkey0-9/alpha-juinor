"""
Quant Research Engine - Alpha Discovery Platform
===================================================

This is the core research workbench for discovering and validating alpha.

NOT a trading module - a RESEARCH platform where hypotheses are:
1. Formulated
2. Tested statistically
3. Validated out-of-sample
4. Deployed if significant

The shift: From feature breadth to empirical depth.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import json
import hashlib

logger = logging.getLogger(__name__)

getcontext().prec = 50


@dataclass
class HypothesisResult:
    """Result of hypothesis testing."""
    hypothesis_id: str
    hypothesis_name: str
    description: str

    # Statistical results
    t_statistic: float
    p_value: float
    mean_return: float
    std_return: float

    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

    # Validation
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    oos_degradation: float  # How much worse OOS vs IS

    # Statistical significance
    is_significant: bool  # p < 0.05
    is_robust: bool  # OOS degradation < 30%

    # Recommendation
    recommendation: str  # DEPLOY, MONITOR, REJECT
    confidence_level: float

    # Metadata
    test_date: datetime
    data_start: datetime
    data_end: datetime
    n_observations: int


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""
    strategy_name: str

    # Period results
    periods: List[Dict[str, Any]]

    # Aggregate metrics
    avg_sharpe: float
    sharpe_std: float
    min_sharpe: float
    max_sharpe: float

    # Consistency
    winning_periods: int
    total_periods: int
    consistency_ratio: float

    # Decay detection
    sharpe_trend: float  # Slope of Sharpe over time
    is_decaying: bool
    decay_rate: float  # Annual decay rate

    # Overall assessment
    overall_grade: str  # A, B, C, D, F
    should_retire: bool


@dataclass
class SignalDefinition:
    """Definition of a trading signal for testing."""
    name: str
    description: str

    # Signal function: takes DataFrame, returns Series of signals
    signal_func: Callable[[pd.DataFrame], pd.Series]

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Lookback required
    lookback_days: int = 60


class StatisticalTester:
    """
    Rigorous statistical testing for alpha hypotheses.

    Tests include:
    - T-test for mean returns
    - Bootstrap confidence intervals
    - Sharpe ratio significance
    - Walk-forward validation
    """

    def __init__(self):
        """Initialize the tester."""
        self.significance_level = 0.05

        logger.info("[RESEARCH] Statistical Tester initialized")

    def test_signal(
        self,
        returns: pd.Series,
        signal: pd.Series,
        name: str = "Signal"
    ) -> HypothesisResult:
        """
        Test if a signal produces statistically significant returns.

        Args:
            returns: Asset returns series
            signal: Signal series (1 for long, -1 for short, 0 for no position)
            name: Signal name
        """
        # Align data
        aligned = pd.concat([returns, signal], axis=1).dropna()
        if len(aligned) < 30:
            return self._null_result(name, "Insufficient data")

        returns = aligned.iloc[:, 0]
        signal = aligned.iloc[:, 1]

        # Calculate strategy returns
        strategy_returns = returns * signal.shift(1)
        strategy_returns = strategy_returns.dropna()

        if len(strategy_returns) < 30:
            return self._null_result(name, "Insufficient data after alignment")

        # Split into IS and OOS
        split_point = int(len(strategy_returns) * 0.7)
        is_returns = strategy_returns.iloc[:split_point]
        oos_returns = strategy_returns.iloc[split_point:]

        # T-test: Is mean return significantly different from zero?
        t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)

        # Calculate metrics
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()

        # Sharpe ratios (annualized)
        is_sharpe = self._calculate_sharpe(is_returns)
        oos_sharpe = self._calculate_sharpe(oos_returns)
        full_sharpe = self._calculate_sharpe(strategy_returns)

        # Sortino ratio
        sortino = self._calculate_sortino(strategy_returns)

        # Max drawdown
        max_dd = self._calculate_max_drawdown(strategy_returns)

        # Win rate
        win_rate = (strategy_returns > 0).mean()

        # Profit factor
        gross_profit = strategy_returns[strategy_returns > 0].sum()
        gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # OOS degradation
        oos_degradation = (is_sharpe - oos_sharpe) / is_sharpe if is_sharpe > 0 else 1.0

        # Significance checks
        is_significant = p_value < self.significance_level
        is_robust = oos_degradation < 0.30 and oos_sharpe > 0.5

        # Recommendation
        if is_significant and is_robust and oos_sharpe > 1.0:
            recommendation = "DEPLOY"
            confidence = min(0.95, 0.7 + (1 - p_value) * 0.25)
        elif is_significant and oos_sharpe > 0.5:
            recommendation = "MONITOR"
            confidence = 0.5 + (1 - p_value) * 0.2
        else:
            recommendation = "REJECT"
            confidence = p_value

        # Generate hypothesis ID
        hypothesis_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        return HypothesisResult(
            hypothesis_id=hypothesis_id,
            hypothesis_name=name,
            description=f"Statistical test of {name} signal",
            t_statistic=float(t_stat),
            p_value=float(p_value),
            mean_return=float(mean_ret * 252),  # Annualized
            std_return=float(std_ret * np.sqrt(252)),
            sharpe_ratio=float(full_sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            profit_factor=float(min(profit_factor, 10)),
            in_sample_sharpe=float(is_sharpe),
            out_of_sample_sharpe=float(oos_sharpe),
            oos_degradation=float(oos_degradation),
            is_significant=is_significant,
            is_robust=is_robust,
            recommendation=recommendation,
            confidence_level=confidence,
            test_date=datetime.utcnow(),
            data_start=strategy_returns.index[0] if hasattr(strategy_returns.index[0], 'isoformat') else datetime.now(),
            data_end=strategy_returns.index[-1] if hasattr(strategy_returns.index[-1], 'isoformat') else datetime.now(),
            n_observations=len(strategy_returns)
        )

    def _calculate_sharpe(self, returns: pd.Series, risk_free: float = 0.04) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 10 or returns.std() == 0:
            return 0
        excess = returns - risk_free / 252
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def _calculate_sortino(self, returns: pd.Series, risk_free: float = 0.04) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) < 10:
            return 0
        excess = returns - risk_free / 252
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 10.0  # No downside
        return float(excess.mean() / downside.std() * np.sqrt(252))

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) < 2:
            return 0
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return float(drawdown.min())

    def _null_result(self, name: str, reason: str) -> HypothesisResult:
        """Return null result when testing fails."""
        return HypothesisResult(
            hypothesis_id="NULL",
            hypothesis_name=name,
            description=f"Test failed: {reason}",
            t_statistic=0,
            p_value=1.0,
            mean_return=0,
            std_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            in_sample_sharpe=0,
            out_of_sample_sharpe=0,
            oos_degradation=1.0,
            is_significant=False,
            is_robust=False,
            recommendation="REJECT",
            confidence_level=0,
            test_date=datetime.utcnow(),
            data_start=datetime.utcnow(),
            data_end=datetime.utcnow(),
            n_observations=0
        )


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.

    Tests strategies across multiple time periods to detect:
    - Performance consistency
    - Strategy decay
    - Regime dependence
    """

    def __init__(self):
        """Initialize the analyzer."""
        self.min_periods = 5
        self.optimization_pct = 0.70  # 70% train, 30% test

        logger.info("[RESEARCH] Walk-Forward Analyzer initialized")

    def analyze(
        self,
        strategy_returns: pd.Series,
        n_periods: int = 10,
        strategy_name: str = "Strategy"
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis on strategy returns.

        Splits data into n_periods and tests each period.
        """
        if len(strategy_returns) < 250:
            return self._null_result(strategy_name, "Insufficient data")

        # Calculate period length
        total_days = len(strategy_returns)
        period_length = total_days // n_periods

        if period_length < 20:
            n_periods = total_days // 20
            period_length = 20

        periods = []
        sharpes = []

        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = min((i + 1) * period_length, total_days)

            period_returns = strategy_returns.iloc[start_idx:end_idx]

            if len(period_returns) < 10:
                continue

            # Calculate period metrics
            sharpe = self._calculate_sharpe(period_returns)
            total_return = (1 + period_returns).prod() - 1
            win_rate = (period_returns > 0).mean()

            periods.append({
                "period": i + 1,
                "start": period_returns.index[0] if hasattr(period_returns.index[0], 'isoformat') else f"Day {start_idx}",
                "end": period_returns.index[-1] if hasattr(period_returns.index[-1], 'isoformat') else f"Day {end_idx}",
                "sharpe": sharpe,
                "return": total_return,
                "win_rate": win_rate,
                "n_days": len(period_returns)
            })
            sharpes.append(sharpe)

        if not sharpes:
            return self._null_result(strategy_name, "No valid periods")

        sharpes = np.array(sharpes)

        # Calculate aggregate metrics
        avg_sharpe = float(np.mean(sharpes))
        sharpe_std = float(np.std(sharpes))

        # Consistency
        winning_periods = int((sharpes > 0).sum())
        consistency = winning_periods / len(sharpes)

        # Decay detection: regress Sharpe on period number
        x = np.arange(len(sharpes))
        if len(sharpes) >= 3:
            slope, _, _, _, _ = stats.linregress(x, sharpes)
            sharpe_trend = float(slope)

            # Annualized decay rate
            periods_per_year = 252 / period_length
            decay_rate = float(slope * periods_per_year)
        else:
            sharpe_trend = 0
            decay_rate = 0

        # Is decaying? (Significant negative trend)
        is_decaying = sharpe_trend < -0.05 and sharpes[-1] < sharpes[0] * 0.7

        # Should retire?
        should_retire = (
            is_decaying or
            avg_sharpe < 0.5 or
            consistency < 0.5 or
            sharpes[-1] < 0
        )

        # Overall grade
        if avg_sharpe >= 1.5 and consistency >= 0.8 and not is_decaying:
            grade = "A"
        elif avg_sharpe >= 1.0 and consistency >= 0.7:
            grade = "B"
        elif avg_sharpe >= 0.5 and consistency >= 0.6:
            grade = "C"
        elif avg_sharpe > 0:
            grade = "D"
        else:
            grade = "F"

        return WalkForwardResult(
            strategy_name=strategy_name,
            periods=periods,
            avg_sharpe=avg_sharpe,
            sharpe_std=sharpe_std,
            min_sharpe=float(np.min(sharpes)),
            max_sharpe=float(np.max(sharpes)),
            winning_periods=winning_periods,
            total_periods=len(periods),
            consistency_ratio=consistency,
            sharpe_trend=sharpe_trend,
            is_decaying=is_decaying,
            decay_rate=decay_rate,
            overall_grade=grade,
            should_retire=should_retire
        )

    def _calculate_sharpe(self, returns: pd.Series, risk_free: float = 0.04) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 5 or returns.std() == 0:
            return 0
        excess = returns - risk_free / 252
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def _null_result(self, name: str, reason: str) -> WalkForwardResult:
        """Return null result."""
        return WalkForwardResult(
            strategy_name=name,
            periods=[],
            avg_sharpe=0,
            sharpe_std=0,
            min_sharpe=0,
            max_sharpe=0,
            winning_periods=0,
            total_periods=0,
            consistency_ratio=0,
            sharpe_trend=0,
            is_decaying=True,
            decay_rate=0,
            overall_grade="F",
            should_retire=True
        )


class AlphaDiscoveryPipeline:
    """
    End-to-end pipeline for discovering and validating alpha.

    1. Formulate hypothesis
    2. Generate signal
    3. Test statistically
    4. Validate out-of-sample
    5. Run walk-forward analysis
    6. Make deployment decision
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.tester = StatisticalTester()
        self.wfa = WalkForwardAnalyzer()

        self.hypotheses_tested = 0
        self.hypotheses_accepted = 0

        logger.info(
            "[RESEARCH] Alpha Discovery Pipeline initialized - "
            "EMPIRICAL DEPTH MODE"
        )

    def test_hypothesis(
        self,
        signal_def: SignalDefinition,
        market_data: pd.DataFrame,
        returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Test a complete hypothesis.

        Returns comprehensive analysis.
        """
        self.hypotheses_tested += 1

        # Generate signal
        try:
            signal = signal_def.signal_func(market_data)
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendation": "REJECT"
            }

        # Statistical test
        stat_result = self.tester.test_signal(
            returns, signal, signal_def.name
        )

        # Walk-forward analysis
        aligned = pd.concat([returns, signal], axis=1).dropna()
        if len(aligned) >= 250:
            strategy_returns = aligned.iloc[:, 0] * aligned.iloc[:, 1].shift(1)
            strategy_returns = strategy_returns.dropna()
            wfa_result = self.wfa.analyze(strategy_returns, strategy_name=signal_def.name)
        else:
            wfa_result = self.wfa._null_result(signal_def.name, "Insufficient data")

        # Combined decision
        if stat_result.recommendation == "DEPLOY" and not wfa_result.should_retire:
            final_recommendation = "DEPLOY"
            self.hypotheses_accepted += 1
        elif stat_result.recommendation == "MONITOR" and wfa_result.overall_grade in ["A", "B", "C"]:
            final_recommendation = "MONITOR"
        else:
            final_recommendation = "REJECT"

        return {
            "success": True,
            "signal_name": signal_def.name,
            "statistical_test": {
                "t_statistic": stat_result.t_statistic,
                "p_value": stat_result.p_value,
                "sharpe_ratio": stat_result.sharpe_ratio,
                "is_significant": stat_result.is_significant,
                "is_robust": stat_result.is_robust,
                "oos_sharpe": stat_result.out_of_sample_sharpe,
                "oos_degradation": stat_result.oos_degradation
            },
            "walk_forward": {
                "avg_sharpe": wfa_result.avg_sharpe,
                "consistency": wfa_result.consistency_ratio,
                "is_decaying": wfa_result.is_decaying,
                "decay_rate": wfa_result.decay_rate,
                "grade": wfa_result.overall_grade,
                "should_retire": wfa_result.should_retire
            },
            "recommendation": final_recommendation,
            "confidence": stat_result.confidence_level,
            "periods_analyzed": wfa_result.total_periods
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        acceptance_rate = (
            self.hypotheses_accepted / self.hypotheses_tested
            if self.hypotheses_tested > 0 else 0
        )

        return {
            "hypotheses_tested": self.hypotheses_tested,
            "hypotheses_accepted": self.hypotheses_accepted,
            "acceptance_rate": acceptance_rate
        }


# Singletons
_tester: Optional[StatisticalTester] = None
_wfa: Optional[WalkForwardAnalyzer] = None
_pipeline: Optional[AlphaDiscoveryPipeline] = None


def get_statistical_tester() -> StatisticalTester:
    """Get or create the Statistical Tester."""
    global _tester
    if _tester is None:
        _tester = StatisticalTester()
    return _tester


def get_walk_forward_analyzer() -> WalkForwardAnalyzer:
    """Get or create the Walk-Forward Analyzer."""
    global _wfa
    if _wfa is None:
        _wfa = WalkForwardAnalyzer()
    return _wfa


def get_alpha_pipeline() -> AlphaDiscoveryPipeline:
    """Get or create the Alpha Discovery Pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AlphaDiscoveryPipeline()
    return _pipeline
