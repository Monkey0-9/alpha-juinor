"""
Learning Feedback Loop - Self-Improvement Engine
=================================================

Continuously learns from trading outcomes:
- Tracks decision outcomes
- Recalibrates model weights
- Adjusts strategy parameters
- Rotates strategies based on performance

This enables the system to get smarter over time.
"""

import logging
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecisionOutcome:
    """Record of a trading decision and its outcome."""
    symbol: str
    decision_type: str  # BUY, SELL, HOLD
    signal_strength: float
    confidence: float
    position_size: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    realized_return: Optional[float] = None
    was_correct: Optional[bool] = None
    contributing_models: List[str] = field(default_factory=list)
    regime: str = "UNKNOWN"


@dataclass
class ModelPerformance:
    """Performance tracking for a single model."""
    model_name: str
    predictions: int = 0
    correct_predictions: int = 0
    total_return: float = 0.0
    avg_confidence: float = 0.0
    hit_rate: float = 0.0
    sharpe_contribution: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update(self, correct: bool, return_val: float, confidence: float):
        """Update performance metrics."""
        self.predictions += 1
        if correct:
            self.correct_predictions += 1
        self.total_return += return_val
        self.avg_confidence = (
            (self.avg_confidence * (self.predictions - 1) + confidence)
            / self.predictions
        )
        self.hit_rate = self.correct_predictions / self.predictions
        self.last_updated = datetime.utcnow()


class LearningFeedbackLoop:
    """
    Self-improvement engine that learns from every decision.

    Capabilities:
    - Tracks all decisions and outcomes
    - Updates model weights based on performance
    - Identifies regime-specific model strengths
    - Suggests strategy adjustments

    Data is persisted to enable learning across sessions.
    """

    # Learning parameters
    LEARNING_RATE = 0.1  # How fast to update weights
    MIN_SAMPLES_FOR_LEARNING = 20  # Minimum decisions before adjustments
    DECAY_HALF_LIFE_DAYS = 21  # How quickly old data loses relevance

    # Weight bounds
    MIN_MODEL_WEIGHT = 0.05
    MAX_MODEL_WEIGHT = 0.40

    def __init__(self, persistence_path: str = "runtime/learning"):
        """Initialize the learning engine."""
        self.persistence_path = persistence_path
        os.makedirs(persistence_path, exist_ok=True)

        # In-memory tracking
        self.pending_decisions: Dict[str, DecisionOutcome] = {}
        self.completed_decisions: List[DecisionOutcome] = []
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.model_weights: Dict[str, float] = {}
        self.regime_weights: Dict[str, Dict[str, float]] = {}

        # Load persisted state
        self._load_state()

        logger.info("[LEARNING] Feedback loop initialized")

    def _load_state(self):
        """Load persisted learning state."""
        try:
            state_file = os.path.join(self.persistence_path, "learning_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)

                self.model_weights = state.get("model_weights", {})
                self.regime_weights = state.get("regime_weights", {})

                # Restore model performance
                for name, perf in state.get("model_performance", {}).items():
                    self.model_performance[name] = ModelPerformance(
                        model_name=name,
                        predictions=perf.get("predictions", 0),
                        correct_predictions=perf.get("correct_predictions", 0),
                        total_return=perf.get("total_return", 0.0),
                        avg_confidence=perf.get("avg_confidence", 0.0),
                        hit_rate=perf.get("hit_rate", 0.0),
                        sharpe_contribution=perf.get("sharpe_contribution", 0.0)
                    )

                logger.info(f"[LEARNING] Loaded state with {len(self.model_weights)} model weights")

        except Exception as e:
            logger.warning(f"[LEARNING] Could not load state: {e}")

    def _save_state(self):
        """Persist learning state."""
        try:
            state = {
                "model_weights": self.model_weights,
                "regime_weights": self.regime_weights,
                "model_performance": {
                    name: asdict(perf)
                    for name, perf in self.model_performance.items()
                },
                "last_updated": datetime.utcnow().isoformat()
            }

            state_file = os.path.join(self.persistence_path, "learning_state.json")
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"[LEARNING] Could not save state: {e}")

    def record_decision(
        self,
        symbol: str,
        decision_type: str,
        signal_strength: float,
        confidence: float,
        position_size: float,
        entry_price: float,
        contributing_models: List[str],
        regime: str = "UNKNOWN"
    ) -> str:
        """
        Record a new trading decision.

        Returns:
            decision_id for later outcome recording
        """
        decision = DecisionOutcome(
            symbol=symbol,
            decision_type=decision_type,
            signal_strength=signal_strength,
            confidence=confidence,
            position_size=position_size,
            entry_price=entry_price,
            contributing_models=contributing_models,
            regime=regime
        )

        decision_id = f"{symbol}_{datetime.utcnow().isoformat()}"
        self.pending_decisions[decision_id] = decision

        logger.debug(f"[LEARNING] Recorded decision {decision_id}")
        return decision_id

    def record_outcome(
        self,
        decision_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None
    ) -> Optional[DecisionOutcome]:
        """
        Record the outcome of a decision.

        This triggers learning updates.
        """
        if decision_id not in self.pending_decisions:
            logger.warning(f"[LEARNING] Unknown decision: {decision_id}")
            return None

        decision = self.pending_decisions.pop(decision_id)
        decision.exit_price = exit_price
        decision.exit_time = exit_time or datetime.utcnow()

        # Calculate return
        if decision.entry_price > 0:
            if decision.decision_type in ["BUY", "STRONG_BUY"]:
                decision.realized_return = (
                    (exit_price - decision.entry_price) / decision.entry_price
                )
            elif decision.decision_type in ["SELL", "STRONG_SELL"]:
                decision.realized_return = (
                    (decision.entry_price - exit_price) / decision.entry_price
                )
            else:
                decision.realized_return = 0.0
        else:
            decision.realized_return = 0.0

        # Determine if decision was correct
        if decision.decision_type in ["BUY", "STRONG_BUY"]:
            decision.was_correct = exit_price > decision.entry_price
        elif decision.decision_type in ["SELL", "STRONG_SELL"]:
            decision.was_correct = exit_price < decision.entry_price
        else:
            decision.was_correct = abs(decision.realized_return) < 0.01

        self.completed_decisions.append(decision)

        # Update model performance
        self._update_model_performance(decision)

        # Trigger learning if enough samples
        if len(self.completed_decisions) >= self.MIN_SAMPLES_FOR_LEARNING:
            self._run_learning_update()

        # Persist
        self._save_state()

        logger.info(
            f"[LEARNING] Outcome recorded: {decision.symbol} "
            f"{'✓' if decision.was_correct else '✗'} "
            f"return={decision.realized_return:.2%}"
        )

        return decision

    def _update_model_performance(self, decision: DecisionOutcome):
        """Update performance tracking for contributing models."""
        for model_name in decision.contributing_models:
            if model_name not in self.model_performance:
                self.model_performance[model_name] = ModelPerformance(
                    model_name=model_name
                )

            self.model_performance[model_name].update(
                correct=decision.was_correct or False,
                return_val=decision.realized_return or 0.0,
                confidence=decision.confidence
            )

    def _run_learning_update(self):
        """
        Run learning update using Exponentiated Gradient.

        w_{t+1,i} = w_{t,i} * exp(η * g_{t,i}) / Z

        Where:
        - w = weight
        - η = learning rate
        - g = gradient (performance signal)
        - Z = normalization constant
        """
        logger.info("[LEARNING] Running weight update...")

        # Calculate performance gradients
        gradients = {}
        for name, perf in self.model_performance.items():
            if perf.predictions >= 5:  # Minimum samples
                # Gradient based on hit rate and return
                gradient = (
                    (perf.hit_rate - 0.5) * 0.5 +  # Hit rate component
                    perf.total_return / max(1, perf.predictions) * 10  # Return component
                )
                gradients[name] = gradient

        if not gradients:
            return

        # Initialize weights if needed
        if not self.model_weights:
            n = len(gradients)
            self.model_weights = {name: 1.0 / n for name in gradients}

        # Exponentiated gradient update
        new_weights = {}
        normalization = 0.0

        for name, gradient in gradients.items():
            current_weight = self.model_weights.get(name, 0.1)
            new_weight = current_weight * np.exp(self.LEARNING_RATE * gradient)
            new_weights[name] = new_weight
            normalization += new_weight

        # Normalize and apply bounds
        for name in new_weights:
            new_weights[name] /= normalization
            new_weights[name] = max(
                self.MIN_MODEL_WEIGHT,
                min(self.MAX_MODEL_WEIGHT, new_weights[name])
            )

        # Re-normalize after bound application
        total = sum(new_weights.values())
        for name in new_weights:
            new_weights[name] /= total

        # Log weight changes
        for name in new_weights:
            old = self.model_weights.get(name, 0)
            new = new_weights[name]
            if abs(new - old) > 0.01:
                direction = "↑" if new > old else "↓"
                logger.info(f"[LEARNING] {name}: {old:.2%} {direction} {new:.2%}")

        self.model_weights = new_weights
        self._save_state()

    def get_model_weights(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get current model weights, optionally regime-specific.

        Args:
            regime: Market regime for regime-specific weights

        Returns:
            Dictionary of model -> weight
        """
        if regime and regime in self.regime_weights:
            return self.regime_weights[regime]
        return self.model_weights

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        total_decisions = len(self.completed_decisions)

        if total_decisions == 0:
            return {"status": "No completed decisions yet"}

        correct = sum(1 for d in self.completed_decisions if d.was_correct)
        total_return = sum(
            d.realized_return or 0 for d in self.completed_decisions
        )

        # Best and worst models
        sorted_models = sorted(
            self.model_performance.items(),
            key=lambda x: x[1].hit_rate,
            reverse=True
        )

        return {
            "total_decisions": total_decisions,
            "correct_decisions": correct,
            "hit_rate": correct / total_decisions,
            "total_return": total_return,
            "avg_return": total_return / total_decisions,
            "best_models": [
                {"name": m.model_name, "hit_rate": m.hit_rate}
                for _, m in sorted_models[:3]
            ],
            "worst_models": [
                {"name": m.model_name, "hit_rate": m.hit_rate}
                for _, m in sorted_models[-3:]
            ],
            "current_weights": self.model_weights
        }

    def get_regime_insights(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by regime."""
        regime_stats: Dict[str, Dict[str, Any]] = {}

        for decision in self.completed_decisions:
            regime = decision.regime
            if regime not in regime_stats:
                regime_stats[regime] = {
                    "count": 0,
                    "correct": 0,
                    "total_return": 0.0
                }

            regime_stats[regime]["count"] += 1
            if decision.was_correct:
                regime_stats[regime]["correct"] += 1
            regime_stats[regime]["total_return"] += decision.realized_return or 0

        # Calculate hit rates
        for regime in regime_stats:
            stats = regime_stats[regime]
            stats["hit_rate"] = stats["correct"] / stats["count"]
            stats["avg_return"] = stats["total_return"] / stats["count"]

        return regime_stats

    def suggest_adjustments(self) -> List[str]:
        """
        Generate suggestions for system improvement.

        Returns:
            List of actionable suggestions
        """
        suggestions = []

        report = self.get_performance_report()

        if isinstance(report, dict) and "status" in report:
            return ["Collect more data before generating suggestions"]

        # Hit rate suggestions
        if report["hit_rate"] < 0.45:
            suggestions.append(
                "⚠️ Hit rate below 45% - consider reducing position sizes"
            )

        if report["hit_rate"] > 0.60:
            suggestions.append(
                "✓ Hit rate above 60% - may increase position sizes"
            )

        # Model-specific suggestions
        for model_perf in report.get("worst_models", []):
            if model_perf["hit_rate"] < 0.40:
                suggestions.append(
                    f"Consider reducing weight of {model_perf['name']} "
                    f"(hit rate: {model_perf['hit_rate']:.0%})"
                )

        # Regime insights
        regime_stats = self.get_regime_insights()
        for regime, stats in regime_stats.items():
            if stats["hit_rate"] < 0.40:
                suggestions.append(
                    f"Review strategy for {regime} regime "
                    f"(hit rate: {stats['hit_rate']:.0%})"
                )

        return suggestions if suggestions else ["System performing within expectations"]

    def reset_learning(self):
        """Reset all learning state (use with caution)."""
        self.pending_decisions = {}
        self.completed_decisions = []
        self.model_performance = {}
        self.model_weights = {}
        self.regime_weights = {}
        self._save_state()
        logger.warning("[LEARNING] All learning state has been reset")


# Singleton instance
_feedback_loop: Optional[LearningFeedbackLoop] = None


def get_feedback_loop() -> LearningFeedbackLoop:
    """Get or create the global Learning Feedback Loop."""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = LearningFeedbackLoop()
    return _feedback_loop
