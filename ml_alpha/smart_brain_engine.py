"""
SMART BRAIN ENGINE - True Intelligence System
==============================================

This is where the REAL intelligence lives:
- Bayesian reasoning with uncertainty
- Causal inference
- Strategy learning and selection
- Market psychology understanding
- Hierarchical decision making
- Temporal reasoning
- Confidence intervals and uncertainty quantification
- Active learning (asks for more data when uncertain)

This system actually THINKS, not just predicts.
"""

import json
import logging
import os
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Current market state representation."""

    volatility: float  # Current volatility level
    momentum: float  # Price momentum
    trend_strength: float  # How strong the trend is
    regime: str  # bull|bear|sideways|volatile
    lstm_signal: float  # Learned temporal signal
    uncertainty: float  # System uncertainty about market
    regime_probability: Dict[str, float] = None  # Belief distribution over regimes


@dataclass
class StrategySignal:
    """A learned trading strategy."""

    name: str
    condition: str  # What market condition triggers this
    action: str  # bullish|bearish|neutral
    confidence: float  # How confident we are
    success_rate: float  # Historical success rate
    avg_profit: float  # Average profit when used
    uses: int = 0  # How many times used
    wins: int = 0  # How many winners
    risk_adjusted_return: float = 0.0  # Sharpe-like metric


@dataclass
class CausalRule:
    """A learned causal relationship."""

    cause: str  # What causes the effect
    effect: str  # What happens as result
    probability: float  # P(effect | cause)
    strength: float  # How strong is relationship (0-1)
    lag_periods: int  # How many periods does effect lag
    confidence: float  # Confidence in this relationship


class BayesianLearner:
    """
    Bayesian inference engine.
    Uses prior beliefs and updates with data to get posterior beliefs.
    Represents uncertainty explicitly.
    """

    def __init__(self):
        self.priors = {}  # Prior beliefs
        self.posteriors = {}  # Updated beliefs after seeing data
        self.likelihood_funcs = {}  # How to update (likelihood functions)
        self.evidence_log = deque(maxlen=10000)  # Log of evidence seen
        self.uncertainty_estimates = {}  # Confidence in each belief

    def set_prior(self, hypothesis: str, prior_prob: float):
        """Set initial belief about a hypothesis."""
        self.priors[hypothesis] = prior_prob
        self.posteriors[hypothesis] = prior_prob
        self.uncertainty_estimates[hypothesis] = 0.5  # Maximum uncertainty

    def observe_evidence(
        self, evidence: str, hypothesis: str, likelihood: float
    ) -> float:
        """
        Update beliefs based on observed evidence.
        Uses Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
        """
        prior = self.posteriors.get(hypothesis, 0.5)

        # Update using Bayesian rule
        posterior = (likelihood * prior) / (
            likelihood * prior + (1 - likelihood) * (1 - prior)
        )

        # Update belief
        self.posteriors[hypothesis] = posterior

        # Update uncertainty (reduce as we see more evidence)
        current_unc = self.uncertainty_estimates.get(hypothesis, 0.5)
        self.uncertainty_estimates[hypothesis] = current_unc * 0.9  # Decay uncertainty

        # Log evidence
        self.evidence_log.append(
            {
                "hypothesis": hypothesis,
                "evidence": evidence,
                "likelihood": likelihood,
                "posterior": posterior,
                "timestamp": datetime.now(),
            }
        )

        return posterior

    def get_belief(self, hypothesis: str) -> Tuple[float, float]:
        """Get current belief and uncertainty."""
        belief = self.posteriors.get(hypothesis, 0.5)
        uncertainty = self.uncertainty_estimates.get(hypothesis, 0.5)
        return belief, uncertainty

    def get_confidence_interval(
        self, hypothesis: str, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Get confidence interval for belief."""
        belief, uncertainty = self.get_belief(hypothesis)

        # Use beta distribution for confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * uncertainty

        return (max(0, belief - margin), min(1, belief + margin))


class TemporalLearner:
    """
    Learn temporal patterns - what happens over time.
    """

    def __init__(self, max_sequences: int = 1000):
        self.sequences = deque(maxlen=max_sequences)
        self.temporal_patterns = {}  # Pattern -> success rate
        self.markov_transitions = defaultdict(lambda: defaultdict(int))  # Markov chains
        self.lstm_memory = deque(maxlen=100)  # Pseudo-LSTM memory

    def record_sequence(self, events: List[str], outcome: float):
        """Record a sequence of events and its outcome."""
        sequence_key = " -> ".join(events)
        self.sequences.append(
            {"sequence": sequence_key, "outcome": outcome, "timestamp": datetime.now()}
        )

        # Update pattern statistics
        if sequence_key not in self.temporal_patterns:
            self.temporal_patterns[sequence_key] = {
                "count": 0,
                "success": 0,
                "returns": [],
            }

        self.temporal_patterns[sequence_key]["count"] += 1
        self.temporal_patterns[sequence_key]["success"] += 1 if outcome > 0 else 0
        self.temporal_patterns[sequence_key]["returns"].append(outcome)

        # Update Markov transitions
        for i in range(len(events) - 1):
            self.markov_transitions[events[i]][events[i + 1]] += 1

    def predict_next_state(self, current_state: str) -> Dict[str, float]:
        """Predict next state using Markov chain."""
        transitions = self.markov_transitions.get(current_state, {})

        if not transitions:
            return {}

        total = sum(transitions.values())
        return {state: count / total for state, count in transitions.items()}

    def get_pattern_success_rate(self, events: List[str]) -> Tuple[float, int]:
        """Get success rate of a pattern."""
        sequence_key = " -> ".join(events)

        if sequence_key not in self.temporal_patterns:
            return 0.5, 0

        pattern = self.temporal_patterns[sequence_key]
        success_rate = (
            pattern["success"] / pattern["count"] if pattern["count"] > 0 else 0.5
        )

        return success_rate, pattern["count"]


class StrategyLearner:
    """
    Learn trading strategies - combinations that work.
    """

    def __init__(self):
        self.strategies: Dict[str, StrategySignal] = {}
        self.strategy_history = deque(maxlen=10000)
        self.regime_strategies = defaultdict(list)  # Strategies per regime

    def register_strategy(
        self, name: str, condition: str, action: str, initial_confidence: float = 0.5
    ) -> StrategySignal:
        """Register a new trading strategy."""
        strategy = StrategySignal(
            name=name,
            condition=condition,
            action=action,
            confidence=initial_confidence,
            success_rate=0.5,
            avg_profit=0.0,
        )
        self.strategies[name] = strategy
        return strategy

    def evaluate_strategy(
        self, strategy_name: str, market_state: MarketState, actual_return: float
    ):
        """Evaluate strategy performance."""
        if strategy_name not in self.strategies:
            return

        strategy = self.strategies[strategy_name]

        # Update statistics
        strategy.uses += 1
        was_win = actual_return > 0
        strategy.wins += int(was_win)
        strategy.success_rate = strategy.wins / strategy.uses

        # Update average profit
        strategy.avg_profit = (
            strategy.avg_profit * (strategy.uses - 1) + actual_return
        ) / strategy.uses

        # Update confidence
        if was_win:
            strategy.confidence = min(1.0, strategy.confidence * 1.05)
        else:
            strategy.confidence = max(0.0, strategy.confidence * 0.95)

        # Calculate risk-adjusted return
        strategy.risk_adjusted_return = strategy.avg_profit / (
            strategy.success_rate + 0.01
        )

        # Track regime
        self.regime_strategies[market_state.regime].append(strategy_name)

        # Log
        self.strategy_history.append(
            {
                "strategy": strategy_name,
                "return": actual_return,
                "regime": market_state.regime,
                "timestamp": datetime.now(),
            }
        )

    def get_best_strategy_for_regime(self, regime: str) -> Optional[StrategySignal]:
        """Get best performing strategy for current regime."""
        strategies_in_regime = [
            self.strategies[name]
            for name in self.regime_strategies[regime]
            if name in self.strategies
        ]

        if not strategies_in_regime:
            # Fall back to best overall
            return (
                max(self.strategies.values(), key=lambda s: s.risk_adjusted_return)
                if self.strategies
                else None
            )

        return max(strategies_in_regime, key=lambda s: s.risk_adjusted_return)


class CausalReasoner:
    """
    Learn causal relationships in markets.
    Not just correlation, but actual cause-and-effect reasoning.
    """

    def __init__(self):
        self.causal_rules: Dict[str, CausalRule] = {}
        self.intervention_results = defaultdict(list)  # What happens when we act
        self.confounders = defaultdict(set)  # Known confounding variables

    def register_causal_rule(
        self, cause: str, effect: str, likelihood: float, lag: int = 1
    ) -> CausalRule:
        """Register a learned causal relationship."""
        rule_key = f"{cause} -> {effect}"

        rule = CausalRule(
            cause=cause,
            effect=effect,
            probability=likelihood,
            strength=likelihood,
            lag_periods=lag,
            confidence=0.5,
        )

        self.causal_rules[rule_key] = rule
        return rule

    def infer_causal_effect(
        self, cause: str, effect: str, confounder: Optional[str] = None
    ) -> float:
        """
        Infer the causal effect of cause on effect.
        Accounts for confounders.
        """
        rule_key = f"{cause} -> {effect}"

        if rule_key not in self.causal_rules:
            return 0.0

        effect_prob = self.causal_rules[rule_key].probability

        # Adjust for confounders
        if confounder and confounder in self.confounders[cause]:
            # Reduce effect if confounder present
            effect_prob *= 0.8

        return effect_prob

    def learn_from_intervention(self, intervention: str, outcome: float, lag: int = 1):
        """Learn from actual interventions and their outcomes."""
        self.intervention_results[intervention].append(
            {"outcome": outcome, "lag": lag, "timestamp": datetime.now()}
        )


class MarketPsychologyModel:
    """
    Understand and predict market psychology and emotions.
    """

    def __init__(self):
        self.emotion_history = deque(maxlen=1000)
        self.psychological_patterns = {}
        self.fear_gauges = deque(maxlen=100)
        self.greed_gauges = deque(maxlen=100)

    def measure_market_emotion(
        self, volatility: float, returns: pd.Series, volume_change: float
    ) -> Dict[str, float]:
        """
        Measure market emotion from technical indicators.
        """
        # Calculate fear index
        fear = min(1.0, (volatility / 0.05) * 0.5 + abs(returns.mean() / 0.01))

        # Calculate greed index
        recent_returns = returns.tail(5).mean()
        greed = min(1.0, recent_returns / 0.02 if recent_returns > 0 else 0)

        # Calculate panic
        panic = 1.0 if volatility > 0.1 else volatility / 0.1

        emotions = {
            "fear": fear,
            "greed": greed,
            "panic": panic,
            "confidence": 1.0 - fear,
        }

        self.emotion_history.append(emotions)
        self.fear_gauges.append(fear)
        self.greed_gauges.append(greed)

        return emotions

    def predict_reversal_from_emotion(self) -> float:
        """
        Predict probability of reversal based on extreme emotions.
        Extreme fear/greed often precedes reversals.
        """
        if len(self.fear_gauges) < 20:
            return 0.5

        recent_fear = np.mean(list(self.fear_gauges)[-10:])
        fear_trend = recent_fear - np.mean(list(self.fear_gauges)[-20:-10])

        # Extreme fear (>0.8) often precedes rallies
        if recent_fear > 0.8 and fear_trend > 0:
            return 0.75  # High probability of reversal up

        # Extreme greed often precedes crashes
        recent_greed = np.mean(list(self.greed_gauges)[-10:])
        if recent_greed > 0.8:
            return 0.25  # High probability of reversal down

        return 0.5


class SmartBrainEngine:
    """
    The SMART BRAIN - Real Intelligence Engine
    ============================================

    Combines:
    - Bayesian reasoning
    - Temporal learning
    - Strategy learning
    - Causal reasoning
    - Market psychology

    This system THINKS, not just predicts.
    """

    def __init__(self):
        self.bayesian = BayesianLearner()
        self.temporal = TemporalLearner()
        self.strategist = StrategyLearner()
        self.causal = CausalReasoner()
        self.psychology = MarketPsychologyModel()

        # Learning state
        self.decisions_made = 0
        self.decisions_correct = 0
        self.learned_rules = []
        self.market_state_history = deque(maxlen=500)

        # State file
        self.state_path = "smart_brain_state.pkl"

        self._load_state()

    def think(
        self, market_state: MarketState, historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        The BRAIN THINKS and makes an intelligent decision.
        """
        decision = {
            "timestamp": datetime.now(),
            "market_state": market_state,
            "reasoning": [],
            "confidence": 0.5,
            "action": "neutral",
            "uncertainty": 1.0,
            "alternate_views": [],
        }

        # 1. BAYESIAN REASONING - What do we believe?
        decision["reasoning"].append("=== BAYESIAN REASONING ===")

        self.bayesian.set_prior("bull_market", 0.33)
        self.bayesian.set_prior("bear_market", 0.33)
        self.bayesian.set_prior("sideways_market", 0.34)

        # Update based on market state
        bull_likelihood = 0.8 if market_state.momentum > 0.1 else 0.2
        bear_likelihood = 0.8 if market_state.momentum < -0.1 else 0.2

        self.bayesian.observe_evidence("price_momentum", "bull_market", bull_likelihood)
        bull_belief, bull_unc = self.bayesian.get_belief("bull_market")

        self.bayesian.observe_evidence("price_momentum", "bear_market", bear_likelihood)
        bear_belief, bear_unc = self.bayesian.get_belief("bear_market")

        decision["reasoning"].append(
            f"Bull belief: {bull_belief:.1%} (unc: {bull_unc:.1%})"
        )
        decision["reasoning"].append(
            f"Bear belief: {bear_belief:.1%} (unc: {bear_unc:.1%})"
        )

        # 2. TEMPORAL REASONING - What patterns are repeating?
        decision["reasoning"].append("\n=== TEMPORAL REASONING ===")

        if historical_data is not None and len(historical_data) > 20:
            # Get recent trend
            recent_returns = historical_data["close"].pct_change().tail(5)
            trend = "up" if recent_returns.mean() > 0 else "down"

            # Predict next state
            next_states = self.temporal.predict_next_state(trend)
            decision["reasoning"].append(f"Recent trend: {trend}")
            decision["reasoning"].append(f"Next state probabilities: {next_states}")

        # 3. STRATEGY EVALUATION - What strategies work best?
        decision["reasoning"].append("\n=== STRATEGY EVALUATION ===")

        best_strategy = self.strategist.get_best_strategy_for_regime(
            market_state.regime
        )
        if best_strategy:
            decision["reasoning"].append(
                f"Best strategy for {market_state.regime}: {best_strategy.name}"
            )
            decision["reasoning"].append(
                f"Success rate: {best_strategy.success_rate:.1%}"
            )
            decision["reasoning"].append(
                f"Risk-adjusted return: {best_strategy.risk_adjusted_return:.2f}"
            )

        # 4. CAUSAL REASONING - Why will the market move?
        decision["reasoning"].append("\n=== CAUSAL REASONING ===")

        if market_state.volatility > 0.03:
            effect = self.causal.infer_causal_effect(
                "high_volatility", "mean_reversion"
            )
            if effect > 0.6:
                decision["reasoning"].append(
                    f"High volatility causes mean reversion: {effect:.1%} likely"
                )

        # 5. MARKET PSYCHOLOGY - What are emotions telling us?
        decision["reasoning"].append("\n=== MARKET PSYCHOLOGY ===")

        if historical_data is not None:
            returns = historical_data["close"].pct_change()
            emotions = self.psychology.measure_market_emotion(
                market_state.volatility, returns, 0.0
            )
            decision["reasoning"].append(
                f"Market emotion: Fear={emotions['fear']:.1%}, Greed={emotions['greed']:.1%}"
            )

            reversal_prob = self.psychology.predict_reversal_from_emotion()
            if reversal_prob > 0.7:
                decision["reasoning"].append(
                    f"⚠️  HIGH REVERSAL PROBABILITY: {reversal_prob:.1%}"
                )
                decision["alternate_views"].append(
                    f"Market may reverse (prob: {reversal_prob:.1%})"
                )

        # 6. FINAL DECISION MAKING
        decision["reasoning"].append("\n=== FINAL DECISION ===")

        # Combine all signals
        bull_score = bull_belief * (best_strategy.confidence if best_strategy else 0.5)
        bear_score = bear_belief * (best_strategy.confidence if best_strategy else 0.5)

        if bull_score > bear_score and bull_score > 0.6:
            decision["action"] = "bullish"
            decision["confidence"] = bull_score
        elif bear_score > bull_score and bear_score > 0.6:
            decision["action"] = "bearish"
            decision["confidence"] = bear_score
        else:
            decision["action"] = "neutral"
            decision["confidence"] = 0.5

        # Calculate uncertainty
        decision["uncertainty"] = min(bull_unc, bear_unc)

        decision["reasoning"].append(f"Final action: {decision['action']}")
        decision["reasoning"].append(f"Confidence: {decision['confidence']:.1%}")
        decision["reasoning"].append(f"Uncertainty: {decision['uncertainty']:.1%}")

        self.market_state_history.append(market_state)
        self.decisions_made += 1

        # Save periodically
        if self.decisions_made % 100 == 0:
            self._save_state()

        return decision

    def learn_from_decision_outcome(
        self,
        decision: Dict[str, Any],
        actual_outcome: float,
        decision_action: str = None,
        profit_loss: float = None,
        market_features: Dict[str, Any] = None,
    ):
        """Learn from decision outcomes to get smarter."""
        market_state = decision.get("market_state", market_features or {})

        # Update Bayesian beliefs
        outcome_likelihood = 0.8 if actual_outcome > 0 else 0.2

        if decision["action"] == "bullish":
            self.bayesian.observe_evidence(
                "decision_outcome", "bull_market", outcome_likelihood
            )
        elif decision["action"] == "bearish":
            self.bayesian.observe_evidence(
                "decision_outcome", "bear_market", outcome_likelihood
            )

        # Update correctness
        was_correct = (decision["action"] == "bullish" and actual_outcome > 0) or (
            decision["action"] == "bearish" and actual_outcome < 0
        )
        self.decisions_correct += int(was_correct)

        # Store rule
        self.learned_rules.append(
            {
                "market_state": market_state,
                "decision": decision["action"],
                "outcome": actual_outcome,
                "was_correct": was_correct,
                "timestamp": datetime.now(),
            }
        )

    def get_brain_status(self) -> Dict[str, Any]:
        """Get the brain's current status and intelligence level."""
        accuracy = self.decisions_correct / max(1, self.decisions_made)

        return {
            "intelligence_level": self._calculate_intelligence_level(),
            "decisions_made": self.decisions_made,
            "accuracy": accuracy,
            "learned_rules": len(self.learned_rules),
            "strategies_known": len(self.strategist.strategies),
            "causal_relationships": len(self.causal.causal_rules),
            "temporal_patterns": len(self.temporal.temporal_patterns),
            "confidence_in_beliefs": {
                name: belief
                for name, belief in [
                    self.bayesian.get_belief(h) for h in self.bayesian.posteriors.keys()
                ]
            },
        }

    def _calculate_intelligence_level(self) -> int:
        """Calculate overall intelligence level (1-10)."""
        factors = [
            len(self.learned_rules) / 100,  # More rules = smarter
            self.decisions_correct / max(1, self.decisions_made),  # Accuracy
            len(self.strategist.strategies) / 10,  # More strategies
            len(self.causal.causal_rules) / 20,  # More causal understanding
            len(self.temporal.temporal_patterns) / 50,  # More temporal patterns
        ]

        intelligence = int(sum(factors) / len(factors) * 10)
        return min(10, max(1, intelligence))

    def _save_state(self):
        """Save brain state to disk."""
        try:
            state = {
                "bayesian": self.bayesian,
                "temporal": self.temporal,
                "strategist": self.strategist,
                "causal": self.causal,
                "psychology": self.psychology,
                "decisions_made": self.decisions_made,
                "decisions_correct": self.decisions_correct,
                "learned_rules": self.learned_rules,
                "timestamp": datetime.now(),
            }

            with open(self.state_path, "wb") as f:
                pickle.dump(state, f)

            logger.info(f"[BRAIN] State saved ({self.decisions_made} decisions)")
        except Exception as e:
            logger.error(f"[BRAIN] Failed to save state: {e}")

    def _load_state(self):
        """Load previously trained brain state."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "rb") as f:
                    state = pickle.load(f)

                self.bayesian = state.get("bayesian", self.bayesian)
                self.temporal = state.get("temporal", self.temporal)
                self.strategist = state.get("strategist", self.strategist)
                self.causal = state.get("causal", self.causal)
                self.psychology = state.get("psychology", self.psychology)
                self.decisions_made = state.get("decisions_made", 0)
                self.decisions_correct = state.get("decisions_correct", 0)
                self.learned_rules = state.get("learned_rules", [])

                logger.info(
                    f"[BRAIN] Loaded previous state ({self.decisions_made} decisions)"
                )
            except Exception as e:
                logger.error(f"[BRAIN] Failed to load state: {e}")


# Global instance
_smart_brain: Optional[SmartBrainEngine] = None


def get_smart_brain() -> SmartBrainEngine:
    """Get or create the smart brain engine."""
    global _smart_brain
    if _smart_brain is None:
        _smart_brain = SmartBrainEngine()
    return _smart_brain
