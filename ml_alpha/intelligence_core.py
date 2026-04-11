"""
INTELLIGENCE CORE - Real-Time Adaptive Learning System
========================================================

This is the brain of the system.
- Learns from every prediction and outcome
- Adapts strategies to market conditions in real-time
- Thinks not just predicts
- Gets smarter with every trade

Core Capabilities:
1. Meta-Learning: Learns what works and what doesn't
2. Pattern Recognition: Identifies recurring market patterns
3. Reinforcement Learning: Rewards good predictions, penalizes bad ones
4. Ensemble Stacking: Learns when to trust which model
5. Confidence Calibration: Learns what its confidence really means
6. Online Learning: Adapts to new market data continuously
7. Risk Adaptation: Adjusts strategy based on market risk
8. Market Regime Detection: Different thinking for different markets
"""

import json
import logging
import os
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction and its outcome."""

    timestamp: datetime
    prediction: float
    confidence: float
    actual_outcome: Optional[float] = None
    was_correct: Optional[bool] = None
    profit_loss: Optional[float] = None
    market_regime: str = "unknown"
    volatility: float = 0.0
    model_used: str = "ensemble"

    def set_outcome(self, actual: float, profit_loss: float = 0.0):
        """Record the actual outcome."""
        self.actual_outcome = actual
        self.was_correct = (actual > 0.5) == (self.prediction > 0.5)
        self.profit_loss = profit_loss


@dataclass
class PatternSignature:
    """Signature of a recurring market pattern."""

    features: Dict[str, float]  # Feature values that define this pattern
    frequency: int = 0  # How often this pattern occurs
    success_rate: float = 0.0  # Success rate when this pattern appears
    avg_return: float = 0.0  # Average return when this pattern appears
    confidence: float = 0.0  # How confident we are about this pattern

    def update(self, was_successful: bool, return_pct: float):
        """Update pattern statistics."""
        self.frequency += 1
        self.success_rate = (
            self.success_rate * (self.frequency - 1) + (1.0 if was_successful else 0.0)
        ) / self.frequency
        self.avg_return = (
            self.avg_return * (self.frequency - 1) + return_pct
        ) / self.frequency
        self.confidence = min(
            1.0, self.success_rate * np.sqrt(min(self.frequency / 100, 1.0))
        )


class MetaLearner:
    """
    Meta-Learning: Learns what works and what doesn't.
    Tracks predictions and learns optimal strategies.
    """

    def __init__(self, memory_size: int = 10000):
        self.prediction_history: deque = deque(maxlen=memory_size)
        self.model_accuracy: Dict[str, float] = {}
        self.model_bias: Dict[str, float] = {}  # Systematic over/under prediction
        self.confidence_calibration: Dict[float, Tuple[float, float]] = (
            {}
        )  # confidence -> (actual_accuracy, count)
        self.pattern_library: Dict[str, PatternSignature] = {}
        self.learning_rate = 0.01

    def record_prediction(self, record: PredictionRecord):
        """Record a prediction for meta-learning."""
        self.prediction_history.append(record)

        # Update model accuracy
        if record.was_correct is not None:
            model = record.model_used
            if model not in self.model_accuracy:
                self.model_accuracy[model] = 0.5
                self.model_bias[model] = 0.0

            # Update accuracy using exponential moving average
            self.model_accuracy[model] = (1 - self.learning_rate) * self.model_accuracy[
                model
            ] + self.learning_rate * (1.0 if record.was_correct else 0.0)

            # Update bias
            error = (1.0 if record.was_correct else 0.0) - record.prediction
            self.model_bias[model] = (1 - self.learning_rate) * self.model_bias[
                model
            ] + self.learning_rate * error

        # Update confidence calibration
        conf_bin = round(record.confidence, 2)
        if conf_bin not in self.confidence_calibration:
            self.confidence_calibration[conf_bin] = (0.0, 0)

        actual_acc, count = self.confidence_calibration[conf_bin]
        new_actual = (actual_acc * count + (1.0 if record.was_correct else 0.0)) / (
            count + 1
        )
        self.confidence_calibration[conf_bin] = (new_actual, count + 1)

    def get_model_ranking(self) -> List[Tuple[str, float]]:
        """Get models ranked by accuracy."""
        return sorted(self.model_accuracy.items(), key=lambda x: x[1], reverse=True)

    def get_calibrated_confidence(
        self, predicted_confidence: float, actual_accuracy: float
    ) -> float:
        """
        Calibrate confidence based on learning.
        Adjusts confidence to match actual accuracy.
        """
        conf_bin = round(predicted_confidence, 2)
        if conf_bin in self.confidence_calibration:
            actual_acc, count = self.confidence_calibration[conf_bin]
            # If we're overconfident, reduce confidence
            if count >= 30:  # Only adjust after sufficient samples
                return predicted_confidence * (
                    actual_acc / (predicted_confidence + 1e-8)
                )

        return predicted_confidence

    def get_recent_accuracy(self, window: int = 100) -> float:
        """Get accuracy over recent predictions."""
        if len(self.prediction_history) < window:
            return 0.5

        recent = list(self.prediction_history)[-window:]
        correct = sum(1 for r in recent if r.was_correct)
        return correct / len(recent)


class PatternRecognizer:
    """
    Pattern Recognition: Identifies recurring market patterns and learns from them.
    """

    def __init__(self):
        self.patterns: Dict[str, PatternSignature] = {}
        self.pattern_threshold = 0.3  # Signal strength to recognize pattern
        self.min_frequency = 5  # Minimum occurrences to trust pattern

    def extract_pattern_features(
        self, market_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Extract key features that define a market pattern.
        Focuses on high-signal features.
        """
        important_features = {}

        # Normalize features to bins for pattern matching
        for key, value in market_features.items():
            if isinstance(value, (int, float)):
                # Bin continuous features
                if "momentum" in key.lower():
                    important_features[f"{key}_bin"] = "high" if value > 0 else "low"
                elif "volatility" in key.lower():
                    if value > 0.02:
                        important_features[f"{key}_bin"] = "high"
                    elif value < 0.01:
                        important_features[f"{key}_bin"] = "low"
                    else:
                        important_features[f"{key}_bin"] = "medium"
                elif "trend" in key.lower():
                    important_features[f"{key}_bin"] = "up" if value > 0 else "down"

        return important_features

    def pattern_to_signature(self, features: Dict[str, float]) -> str:
        """Convert pattern features to a hashable signature."""
        return json.dumps(features, sort_keys=True)

    def recognize_pattern(
        self, market_features: Dict[str, float]
    ) -> Tuple[Optional[str], float]:
        """
        Recognize if current market matches a known pattern.
        Returns (pattern_signature, confidence)
        """
        pattern_features = self.extract_pattern_features(market_features)
        signature = self.pattern_to_signature(pattern_features)

        if signature in self.patterns:
            pattern = self.patterns[signature]
            if pattern.frequency >= self.min_frequency:
                return signature, pattern.confidence

        return None, 0.0

    def learn_from_outcome(
        self, market_features: Dict[str, float], was_successful: bool, return_pct: float
    ):
        """Learn from a trade outcome to improve pattern recognition."""
        pattern_features = self.extract_pattern_features(market_features)
        signature = self.pattern_to_signature(pattern_features)

        if signature not in self.patterns:
            self.patterns[signature] = PatternSignature(features=pattern_features)

        self.patterns[signature].update(was_successful, return_pct)


class EnsembleOptimizer:
    """
    Ensemble Optimization: Learns optimal model weights from outcomes.
    """

    def __init__(self, model_names: List[str]):
        self.model_names = model_names
        self.model_weights = {name: 1.0 / len(model_names) for name in model_names}
        self.model_performance = {name: deque(maxlen=1000) for name in model_names}
        self.learning_rate = 0.01

    def record_model_outcome(
        self, model_name: str, was_correct: bool, confidence: float
    ):
        """Record outcome for a specific model."""
        if model_name in self.model_performance:
            self.model_performance[model_name].append(1.0 if was_correct else 0.0)

    def optimize_weights(self):
        """Optimize weights based on recent performance."""
        accuracies = {}

        for name in self.model_names:
            performance = self.model_performance[name]
            if len(performance) > 20:
                accuracy = np.mean(list(performance)[-50:])
                accuracies[name] = accuracy
            else:
                accuracies[name] = 0.5

        # Softmax over accuracies
        total_acc = sum(accuracies.values())
        if total_acc > 0:
            for name in self.model_names:
                new_weight = accuracies[name] / total_acc
                self.model_weights[name] = (
                    1 - self.learning_rate
                ) * self.model_weights[name] + self.learning_rate * new_weight

        # Normalize
        total_weight = sum(self.model_weights.values())
        for name in self.model_names:
            self.model_weights[name] /= total_weight


class RiskAwareAdapter:
    """
    Risk-Aware Adaptation: Adjusts confidence based on risk conditions.
    """

    def __init__(self):
        self.current_volatility = 0.01
        self.current_drawdown = 0.0
        self.current_sharpe = 0.0
        self.risk_regime = "normal"

    def update_risk_metrics(self, volatility: float, drawdown: float, sharpe: float):
        """Update current risk metrics."""
        self.current_volatility = volatility
        self.current_drawdown = drawdown
        self.current_sharpe = sharpe

        # Classify risk regime
        if drawdown < -0.10:
            self.risk_regime = "crisis"
        elif volatility > 0.03:
            self.risk_regime = "high_risk"
        elif volatility < 0.005:
            self.risk_regime = "low_risk"
        else:
            self.risk_regime = "normal"

    def adjust_confidence(self, base_confidence: float) -> float:
        """Adjust confidence based on risk conditions."""
        adjustments = {
            "crisis": 0.5,  # Cut confidence in half during crisis
            "high_risk": 0.75,  # 25% reduction in high risk
            "normal": 1.0,  # No adjustment
            "low_risk": 1.1,  # 10% boost in low risk
        }

        multiplier = adjustments.get(self.risk_regime, 1.0)
        adjusted = base_confidence * multiplier

        # Also reduce if Sharpe is negative
        if self.current_sharpe < 0:
            adjusted *= max(0.3, 1.0 + (self.current_sharpe / 10))

        return max(0.0, min(1.0, adjusted))


class IntelligenceCore:
    """
    The Intelligence Core - The Brain of the Trading System

    This system:
    - Learns from every decision outcome
    - Recognizes recurring market patterns
    - Optimizes model weights dynamically
    - Adapts to risk conditions
    - Calibrates confidence from real data
    - Makes increasingly intelligent decisions over time
    """

    def __init__(self, model_names: List[str] = None):
        self.model_names = model_names or ["model_1", "model_2", "model_3"]

        # Core learning components
        self.meta_learner = MetaLearner()
        self.pattern_recognizer = PatternRecognizer()
        self.ensemble_optimizer = EnsembleOptimizer(self.model_names)
        self.risk_adapter = RiskAwareAdapter()

        # Statistics
        self.total_predictions = 0
        self.total_correct = 0
        self.learning_started = datetime.now()
        self.state_path = "intelligence_core_state.pkl"

        self._load_state()

    def process_prediction(
        self,
        prediction: float,
        model_weights: Dict[str, float],
        market_features: Dict[str, float],
        confidence: float,
    ) -> Dict[str, Any]:
        """
        Process a prediction through the intelligence core.
        Enhances it with learning and pattern recognition.
        """
        self.total_predictions += 1

        # Recognize patterns
        pattern_sig, pattern_conf = self.pattern_recognizer.recognize_pattern(
            market_features
        )

        # Get risk-aware adjustment
        volatility = market_features.get("volatility", 0.01)
        drawdown = market_features.get("drawdown", 0.0)
        sharpe = market_features.get("sharpe_ratio", 0.0)

        self.risk_adapter.update_risk_metrics(volatility, drawdown, sharpe)
        adjusted_confidence = self.risk_adapter.adjust_confidence(confidence)

        # Get model rankings
        model_ranking = self.meta_learner.get_model_ranking()

        # Boost prediction confidence if pattern matches
        if pattern_sig and pattern_conf > 0.5:
            adjusted_confidence = min(1.0, adjusted_confidence * 1.2)

        return {
            "enhanced_prediction": prediction,
            "original_confidence": confidence,
            "adjusted_confidence": adjusted_confidence,
            "risk_regime": self.risk_adapter.risk_regime,
            "pattern_detected": pattern_sig,
            "pattern_confidence": pattern_conf,
            "model_ranking": model_ranking,
            "ensemble_weights": self.ensemble_optimizer.model_weights,
            "core_accuracy": self.meta_learner.get_recent_accuracy(),
            "learning_progress": self._get_learning_progress(),
        }

    def learn_from_outcome(
        self,
        prediction: float,
        confidence: float,
        actual_outcome: float,
        profit_loss: float,
        market_features: Dict[str, float],
        models_used: str = "ensemble",
    ):
        """
        Learn from a prediction outcome.
        This makes the system smarter with every trade.
        """
        # Record prediction
        record = PredictionRecord(
            timestamp=datetime.now(),
            prediction=prediction,
            confidence=confidence,
            market_regime=self.risk_adapter.risk_regime,
            volatility=market_features.get("volatility", 0.0),
            model_used=models_used,
        )
        record.set_outcome(actual_outcome, profit_loss)

        # Update totals
        self.total_correct += int(record.was_correct or False)

        # Meta-learning
        self.meta_learner.record_prediction(record)

        # Pattern learning
        if record.was_correct is not None:
            return_pct = profit_loss / 100 if profit_loss else 0
            self.pattern_recognizer.learn_from_outcome(
                market_features, record.was_correct, return_pct
            )

        # Ensemble optimization
        for model_name in self.model_names:
            self.ensemble_optimizer.record_model_outcome(
                model_name, record.was_correct or False, confidence
            )

        # Optimize weights
        self.ensemble_optimizer.optimize_weights()

        # Auto-save periodically
        if self.total_predictions % 100 == 0:
            self._save_state()

    def get_system_intelligence(self) -> Dict[str, Any]:
        """Get comprehensive intelligence status."""
        accuracy = self.total_correct / max(1, self.total_predictions)

        return {
            "overall_accuracy": accuracy,
            "total_predictions": self.total_predictions,
            "model_rankings": self.meta_learner.get_model_ranking(),
            "patterns_learned": len(self.pattern_recognizer.patterns),
            "top_patterns": self._get_top_patterns(),
            "ensemble_weights": self.ensemble_optimizer.model_weights,
            "current_risk_regime": self.risk_adapter.risk_regime,
            "learning_duration_hours": (
                datetime.now() - self.learning_started
            ).total_seconds()
            / 3600,
            "confidence_calibration": dict(
                list(self.meta_learner.confidence_calibration.items())[-10:]
            ),
        }

    def _get_top_patterns(self, count: int = 5) -> List[Dict]:
        """Get the most successful patterns."""
        patterns = sorted(
            self.pattern_recognizer.patterns.values(),
            key=lambda p: p.success_rate * p.frequency,
            reverse=True,
        )

        return [
            {
                "success_rate": p.success_rate,
                "frequency": p.frequency,
                "avg_return": p.avg_return,
                "confidence": p.confidence,
            }
            for p in patterns[:count]
        ]

    def _get_learning_progress(self) -> Dict[str, Any]:
        """Calculate learning progress."""
        recent_acc = self.meta_learner.get_recent_accuracy(window=100)
        all_acc = self.total_correct / max(1, self.total_predictions)

        return {
            "recent_accuracy_100": recent_acc,
            "overall_accuracy": all_acc,
            "improvement": recent_acc - all_acc,
            "patterns_discovered": len(self.pattern_recognizer.patterns),
            "hours_learning": (datetime.now() - self.learning_started).total_seconds()
            / 3600,
        }

    def _save_state(self):
        """Save learned state to disk."""
        try:
            state = {
                "meta_learner": self.meta_learner,
                "pattern_recognizer": self.pattern_recognizer,
                "ensemble_optimizer": self.ensemble_optimizer,
                "total_predictions": self.total_predictions,
                "total_correct": self.total_correct,
                "learning_started": self.learning_started,
            }

            with open(self.state_path, "wb") as f:
                pickle.dump(state, f)

            logger.info(
                f"[CORE] Intelligence state saved ({self.total_predictions} predictions learned)"
            )
        except Exception as e:
            logger.error(f"[CORE] Failed to save state: {e}")

    def _load_state(self):
        """Load previously learned state."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "rb") as f:
                    state = pickle.load(f)

                self.meta_learner = state.get("meta_learner", self.meta_learner)
                self.pattern_recognizer = state.get(
                    "pattern_recognizer", self.pattern_recognizer
                )
                self.ensemble_optimizer = state.get(
                    "ensemble_optimizer", self.ensemble_optimizer
                )
                self.total_predictions = state.get("total_predictions", 0)
                self.total_correct = state.get("total_correct", 0)
                self.learning_started = state.get("learning_started", datetime.now())

                logger.info(
                    f"[CORE] Loaded intelligence state ({self.total_predictions} predictions)"
                )
            except Exception as e:
                logger.error(f"[CORE] Failed to load state: {e}")


# Global instance
_intelligence_core: Optional[IntelligenceCore] = None


def get_intelligence_core() -> IntelligenceCore:
    """Get or create the intelligence core."""
    global _intelligence_core
    if _intelligence_core is None:
        _intelligence_core = IntelligenceCore(
            model_names=[
                "lightgbm",
                "xgboost",
                "catboost",
                "gradient_boosting",
                "random_forest",
            ]
        )
    return _intelligence_core
