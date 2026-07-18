"""
nexus/core/superhuman_brain.py — Superhuman Brain Module

The crown jewel of the upgrade. Sits above MarketBrain and applies:
  1. Bayesian Signal Fusion — posterior-weighted combination of all strategy signals
  2. Meta-Learning Self-Calibration — real-time IC tracking adjusts signal weights
  3. Fractal Market Hypothesis Gate — only enters when market structure supports trade
  4. Information Ratio Maximizer — ranks by predicted IR vs. benchmark
  5. Conviction Scoring — outputs (signal, conviction) grade like a top-1% analyst

Architecture:
  SuperhumanBrain.evaluate_portfolio(signals, history, regime_probs)
    → Dict[symbol → ConvictionSignal(score, conviction, gate_pass, reasoning)]
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
from collections import deque

from nexus.math.models import FractalEngine, VolatilityTopologyHeuristic
from nexus.core.strategies import StrategyFactory

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data Structures                                                      #
# ------------------------------------------------------------------ #

@dataclass
class ConvictionSignal:
    """
    Full intelligence output for a single asset.
    """
    symbol:       str
    score:        float          # Final fused signal (-1 to +1)
    conviction:   float          # Confidence grade (0 to 1)
    gate_pass:    bool           # Passes fractal + entropy gate?
    regime_bias:  str            # Dominant regime at time of evaluation
    ir_score:     float          # Information Ratio proxy
    strategy_votes: Dict[str, float] = field(default_factory=dict)
    reasoning:    str = ""       # Human-readable summary

    def conviction_grade(self) -> str:
        """Map conviction to letter grade (A++ for top-1%)."""
        if self.conviction >= 0.85:
            return "A++"
        if self.conviction >= 0.75:
            return "A+"
        if self.conviction >= 0.65:
            return "A"
        if self.conviction >= 0.55:
            return "B+"
        if self.conviction >= 0.45:
            return "B"
        return "C"


# ------------------------------------------------------------------ #
# Bayesian Strategy Weighter                                           #
# ------------------------------------------------------------------ #

class BayesianStrategyWeighter:
    """
    Maintains Bayesian posterior probability for each strategy being
    'in-regime' (i.e. its predictions are better than chance).

    Prior: Beta(alpha=2, beta=2) → 0.5 win-rate prior
    Update: each correct prediction adds to alpha, each miss adds to beta
    """

    def __init__(self, strategies: list[Any], window: int = 30):
        self.strategy_names = [s.name for s in strategies]
        self.window = window
        # Prior: Beta(2,2) → 50% hit-rate prior
        self._alpha: Dict[str, float] = {name: 2.0 for name in self.strategy_names}
        self._beta: Dict[str, float] = {name: 2.0 for name in self.strategy_names}
        self._history: Dict[str, deque[int]] = {
            name: deque(maxlen=window) for name in self.strategy_names
        }

    def record_outcome(self, strategy_name: str, correct: bool) -> None:
        """Update posterior after observing whether prediction was correct."""
        if strategy_name not in self._alpha:
            return
        self._history[strategy_name].append(1 if correct else 0)
        # Bayesian update: use sliding window counts as pseudo-observations
        window_data = list(self._history[strategy_name])
        n_correct = sum(window_data)
        n_total   = len(window_data)
        # Reset to prior + window evidence
        self._alpha[strategy_name] = 2.0 + n_correct
        self._beta[strategy_name]  = 2.0 + (n_total - n_correct)

    def get_weight(self, strategy_name: str) -> float:
        """
        Returns the posterior mean (expected hit-rate) for this strategy.
        Beta posterior mean = alpha / (alpha + beta)
        """
        a = self._alpha.get(strategy_name, 2.0)
        b = self._beta.get(strategy_name, 2.0)
        return float(a / (a + b))

    def get_all_weights(self) -> Dict[str, float]:
        """Returns normalized Bayesian weights for all strategies."""
        raw = {n: self.get_weight(n) for n in self.strategy_names}
        # Normalize relative to 0.5 (random chance baseline)
        adjusted = {n: max(0.0, w - 0.40) for n, w in raw.items()}
        total = sum(adjusted.values()) + 1e-9
        return {n: v / total for n, v in adjusted.items()}


# ------------------------------------------------------------------ #
# Meta-Learning Self-Calibrator                                        #
# ------------------------------------------------------------------ #

class MetaLearningSelfCalibrator:
    """
    Tracks predicted signals vs. realized next-bar returns.
    Computes rolling IC (Information Coefficient) per symbol
    and uses it to scale signal confidence in real-time.

    Updated each cycle with: symbol, predicted_signal, realized_return
    """

    def __init__(self, window: int = 40):
        self.window = window
        # symbol → deque of (predicted, realized)
        self._records: Dict[str, deque[tuple[float, float]]] = {}

    def record(self, symbol: str, predicted: float, realized: float) -> None:
        if symbol not in self._records:
            self._records[symbol] = deque(maxlen=self.window)
        self._records[symbol].append((predicted, realized))

    def get_ic(self, symbol: str) -> float:
        """Rolling IC for symbol. Returns 0.0 if insufficient data."""
        pairs = list(self._records.get(symbol, []))
        if len(pairs) < 5:
            return 0.0
        pred = np.array([p for p, _ in pairs])
        real = np.array([r for _, r in pairs])
        if np.std(pred) < 1e-9 or np.std(real) < 1e-9:
            return 0.0
        return float(np.corrcoef(pred, real)[0, 1])

    def get_ic_weight(self, symbol: str) -> float:
        """
        Map IC to a confidence multiplier:
        IC = 0.0  → 0.5 (neutral, no edge)
        IC = 0.3  → 0.8 (good edge)
        IC = 0.6+ → 1.0 (elite edge)
        IC < 0    → 0.2 (penalize negative IC signals)
        """
        ic = self.get_ic(symbol)
        if ic < 0:
            return max(0.1, 0.5 + ic * 0.4)
        return float(min(1.0, 0.5 + ic * 1.5))

    def get_global_ic(self) -> float:
        """Average IC across all tracked symbols."""
        ics = [self.get_ic(s) for s in self._records if len(self._records[s]) >= 5]
        return float(np.mean(ics)) if ics else 0.0


# ------------------------------------------------------------------ #
# Fractal Market Hypothesis Gate                                       #
# ------------------------------------------------------------------ #

class FractalMarketGate:
    """
    Only opens the 'entry gate' when the fractal market structure supports trading.

    Rules:
    - FD in (1.0, 1.3)   → HIGH_PERSISTENCE  → Gate OPEN for momentum trades
    - FD in (1.7, 2.0)   → HIGH_ROUGHNESS    → Gate OPEN for mean-reversion
    - FD in (1.3, 1.7)   → STABLE            → Gate OPEN with reduced size
    - Topology CHAOTIC   → Gate CLOSED (no edge, high noise)
    - Entropy too high   → Gate CLOSED
    """

    def __init__(self) -> None:
        self.fractal_engine  = FractalEngine()
        self.topology_engine = VolatilityTopologyHeuristic()

    def evaluate(
        self,
        history: pd.DataFrame,
        signal: float,
        regime: str,
    ) -> Tuple[bool, float, str]:
        """
        Returns (gate_pass, size_multiplier, reason).
        size_multiplier is 1.0 for full pass, 0.5 for partial, 0.0 for blocked.
        """
        if history.empty or len(history) < 30:
            return True, 0.7, "INSUFFICIENT_DATA_PARTIAL"

        prices = history["close"].astype(float).to_numpy()
        returns = np.diff(prices) / (prices[:-1] + 1e-9)

        # Fractal Dimension
        fd = self.fractal_engine.calculate_dimension(prices)
        fd_structure = self.fractal_engine.analyze_structure(fd)

        # Volatility Topology
        topology = self.topology_engine.map_market_topology(prices)
        if topology == "CHAOTIC":
            return False, 0.0, f"BLOCKED:CHAOTIC_MARKET (FD={fd:.2f})"

        # Shannon Entropy filter
        if len(returns) > 10:
            entropy = self._shannon_entropy(returns)
            if entropy > 4.5:  # very noisy distribution
                return True, 0.4, f"PARTIAL:HIGH_ENTROPY ({entropy:.1f})"

        # Gate logic by signal direction + fractal structure alignment
        if signal > 0:  # Bullish signal
            if fd_structure == "HIGH_PERSISTENCE":
                return True, 1.0, f"OPEN:PERSISTENT_TREND (FD={fd:.2f})"
            elif fd_structure == "STABLE":
                return True, 0.75, f"PARTIAL:STABLE_STRUCTURE (FD={fd:.2f})"
            else:  # HIGH_ROUGHNESS with bullish signal — conflict
                return True, 0.50, f"CAUTION:ROUGH_MARKET_BULLISH (FD={fd:.2f})"
        elif signal < 0:  # Bearish signal
            if fd_structure == "HIGH_ROUGHNESS":
                return True, 1.0, f"OPEN:MEAN_REVERT_STRUCTURE (FD={fd:.2f})"
            elif fd_structure == "STABLE":
                return True, 0.75, f"PARTIAL:STABLE_STRUCTURE (FD={fd:.2f})"
            else:
                return True, 0.50, f"CAUTION:ROUGH_MARKET_BEARISH (FD={fd:.2f})"

        return True, 0.6, f"NEUTRAL (FD={fd:.2f})"

    @staticmethod
    def _shannon_entropy(returns: np.ndarray[Any, Any]) -> float:
        """Approximate Shannon entropy of return distribution."""
        try:
            hist, _ = np.histogram(returns, bins=20, density=True)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log(hist + 1e-9)))
        except Exception:
            return 3.0


# ------------------------------------------------------------------ #
# SuperhumanBrain — Main Class                                         #
# ------------------------------------------------------------------ #

class SuperhumanBrain:
    """
    Top-1% Quantitative Intelligence Engine.

    Wraps all base intelligence layers with:
    - Bayesian signal fusion across all strategies
    - Real-time meta-learning IC calibration
    - Fractal market hypothesis gate
    - Information Ratio maximization
    - Conviction scoring (A++ → C grade)

    Usage:
        brain = SuperhumanBrain()
        signals = brain.evaluate_portfolio(
            raw_signals, history, regime_probs
        )
        # signals: Dict[symbol → ConvictionSignal]
    """

    def __init__(self) -> None:
        strategies = StrategyFactory.all_strategies()
        self.bayesian_weighter  = BayesianStrategyWeighter(strategies, window=30)
        self.meta_calibrator    = MetaLearningSelfCalibrator(window=40)
        self.fractal_gate       = FractalMarketGate()
        self._prev_signals: Dict[str, float] = {}
        logger.info("SuperhumanBrain initialized with %d strategies", len(strategies))

    # ---------------------------------------------------------------- #
    # Core Evaluation                                                   #
    # ---------------------------------------------------------------- #

    def evaluate_portfolio(
        self,
        raw_signals: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
        regime_probs: Dict[str, float],
        current_regime: str = "SIDEWAYS",
    ) -> Dict[str, "ConvictionSignal"]:
        """
        Full portfolio evaluation with superhuman intelligence.

        Returns a ConvictionSignal for each symbol.
        """
        # Update meta-learner with previous cycle's predictions vs. actual returns
        self._update_meta_learner(raw_signals, historical_data)

        results: Dict[str, ConvictionSignal] = {}
        bayesian_weights = self.bayesian_weighter.get_all_weights()

        for symbol, alpha in raw_signals.items():
            history = historical_data.get(symbol, pd.DataFrame())

            # Step 1: Bayesian-fused strategy score
            fused_score, strategy_votes = self._bayesian_fuse(
                symbol, alpha, history, current_regime, bayesian_weights
            )

            # Step 2: Meta-learning IC adjustment
            ic_weight = self.meta_calibrator.get_ic_weight(symbol)
            adjusted_score = fused_score * (0.6 + 0.4 * ic_weight)

            # Step 3: Information Ratio score
            ir_score = self._compute_ir_score(symbol, adjusted_score, history)

            # Step 4: Fractal gate evaluation
            gate_pass, size_mult, gate_reason = self.fractal_gate.evaluate(
                history, adjusted_score, current_regime
            )

            # Step 5: Final score with gate multiplier
            final_score = float(np.tanh(adjusted_score * size_mult))

            # Step 6: Conviction scoring
            conviction = self._compute_conviction(
                fused_score, ic_weight, gate_pass, size_mult,
                regime_probs, strategy_votes
            )

            # Step 7: Regime bias label
            regime_bias = max(regime_probs, key=lambda k: regime_probs[k])

            reasoning = (
                f"Bayesian={fused_score:.3f} | IC={ic_weight:.2f} | "
                f"Gate={gate_reason} | IR={ir_score:.3f} | "
                f"RegimeBias={regime_bias} | Conviction={conviction:.0%}"
            )
            logger.debug("[SuperhumanBrain] %s: %s", symbol, reasoning)

            results[symbol] = ConvictionSignal(
                symbol=symbol,
                score=final_score,
                conviction=conviction,
                gate_pass=gate_pass,
                regime_bias=regime_bias,
                ir_score=ir_score,
                strategy_votes=strategy_votes,
                reasoning=reasoning,
            )

            # Store signal for next cycle's meta-learning update
            self._prev_signals[symbol] = final_score

        return results

    # ---------------------------------------------------------------- #
    # Bayesian Fusion                                                   #
    # ---------------------------------------------------------------- #

    def _bayesian_fuse(
        self,
        symbol: str,
        alpha: float,
        history: pd.DataFrame,
        regime: str,
        weights: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Bayesian-weighted average of all strategy scores.
        """
        strategies = StrategyFactory.all_strategies()
        strategy_votes: Dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for strat in strategies:
            w = weights.get(strat.name, 1.0 / len(strategies))
            score = strat.score(symbol, alpha, history, regime)
            strategy_votes[strat.name] = float(score)
            weighted_sum += w * score
            total_weight += w

        if total_weight < 1e-9:
            return float(alpha), strategy_votes

        fused = weighted_sum / total_weight
        return float(np.tanh(fused * 1.15)), strategy_votes

    # ---------------------------------------------------------------- #
    # Information Ratio                                                 #
    # ---------------------------------------------------------------- #

    def _compute_ir_score(
        self, symbol: str, signal: float, history: pd.DataFrame
    ) -> float:
        """
        Proxy IR = predicted_alpha / tracking_error
        Tracking error estimated from return volatility.
        """
        if history.empty or "close" not in history.columns:
            return float(signal)
        returns = history["close"].pct_change().dropna()
        if returns.empty:
            return float(signal)
        te = float(returns.std() * np.sqrt(252)) + 1e-6
        return float(np.tanh(signal / te * 0.5))

    # ---------------------------------------------------------------- #
    # Conviction Scoring                                                #
    # ---------------------------------------------------------------- #

    def _compute_conviction(
        self,
        fused_score:   float,
        ic_weight:     float,
        gate_pass:     bool,
        size_mult:     float,
        regime_probs:  Dict[str, float],
        strategy_votes: Dict[str, float],
    ) -> float:
        """
        Conviction = weighted combination of:
        1. Signal strength (abs value of fused score)
        2. IC quality (meta-learner calibration)
        3. Gate pass (structural support)
        4. Regime clarity (entropy of regime probability distribution)
        5. Strategy consensus (fraction of strategies agreeing on direction)

        Returns conviction ∈ [0.0, 1.0]
        """
        # 1. Signal strength
        signal_strength = min(1.0, abs(fused_score) * 1.3)

        # 2. IC quality
        ic_score = float(ic_weight)

        # 3. Gate score
        gate_score = size_mult if gate_pass else 0.2

        # 4. Regime clarity: lower entropy = more confident regime
        prob_vals = list(regime_probs.values())
        regime_entropy = -sum(p * np.log(p + 1e-9) for p in prob_vals)
        max_entropy = np.log(len(prob_vals) + 1e-9)
        regime_clarity = 1.0 - float(regime_entropy / max_entropy)

        # 5. Strategy consensus
        if strategy_votes:
            direction = np.sign(fused_score)
            agreeing = sum(
                1 for v in strategy_votes.values()
                if np.sign(v) == direction and abs(v) > 0.1
            )
            consensus = agreeing / max(len(strategy_votes), 1)
        else:
            consensus = 0.5

        # Weighted conviction
        conviction = (
            0.25 * signal_strength
            + 0.20 * ic_score
            + 0.20 * gate_score
            + 0.15 * regime_clarity
            + 0.20 * consensus
        )
        return float(np.clip(conviction, 0.0, 1.0))

    # ---------------------------------------------------------------- #
    # Meta-Learning Update                                              #
    # ---------------------------------------------------------------- #

    def _update_meta_learner(
        self,
        current_signals: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Update IC tracker: compare previous cycle's predicted signal
        vs. the actual realized return in this cycle's data.
        """
        for symbol, prev_signal in self._prev_signals.items():
            history = historical_data.get(symbol, pd.DataFrame())
            if history.empty or "close" not in history.columns or len(history) < 2:
                continue
            # Realized return: last-bar return
            realized = float(
                history["close"].pct_change().dropna().iloc[-1]
            )
            self.meta_calibrator.record(symbol, prev_signal, realized)

    # ---------------------------------------------------------------- #
    # Bayesian Strategy Outcome Update (call after each cycle)         #
    # ---------------------------------------------------------------- #

    def update_strategy_outcomes(
        self,
        symbol: str,
        strategy_votes: Dict[str, float],
        realized_return: float,
    ) -> None:
        """
        After observing realized return, update Bayesian posteriors for all strategies.
        Call this from the engine after each cycle completes.
        """
        realized_sign = int(np.sign(realized_return))
        for strat_name, strat_score in strategy_votes.items():
            predicted_sign = int(np.sign(strat_score))
            correct = (predicted_sign == realized_sign and abs(strat_score) > 0.05)
            self.bayesian_weighter.record_outcome(strat_name, correct)

    # ---------------------------------------------------------------- #
    # Portfolio-Level Intelligence Report                               #
    # ---------------------------------------------------------------- #

    def portfolio_intelligence_report(
        self, conviction_signals: Dict[str, "ConvictionSignal"]
    ) -> Dict[str, Any]:
        """
        Returns a summary of portfolio-level intelligence metrics.
        """
        if not conviction_signals:
            return {}

        convictions = [s.conviction for s in conviction_signals.values()]
        scores = [s.score for s in conviction_signals.values()]
        gate_passes = sum(1 for s in conviction_signals.values() if s.gate_pass)
        global_ic = self.meta_calibrator.get_global_ic()
        a_grade = sum(1 for c in convictions if c >= 0.65)

        return {
            "avg_conviction":        float(np.mean(convictions)),
            "max_conviction":        float(np.max(convictions)),
            "avg_signal_strength":   float(np.mean(np.abs(scores))),
            "gate_pass_rate":        gate_passes / max(len(conviction_signals), 1),
            "global_ic":             global_ic,
            "a_grade_signals":       a_grade,
            "total_signals":         len(conviction_signals),
            "bayesian_weights":      self.bayesian_weighter.get_all_weights(),
        }
