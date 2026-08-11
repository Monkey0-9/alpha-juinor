import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class AIEnsembleBrain:
    """
    Superhuman AI Ensemble Brain.
    Coordinates predictions across multiple Machine Learning models,
    weighs them by market regime and hit-rate history, and enforces
    explicit NO_TRADE gating when market noise is high.
    """

    def __init__(self, confidence_gate: float = 0.35):
        self.models: List[Dict[str, Any]] = []
        self.confidence_gate = confidence_gate
        self.performance_history: Dict[str, List[int]] = {}

    def add_model(self, model: Any, name: str = "Model", weight: float = 1.0) -> None:
        self.models.append({
            'name': name,
            'model': model,
            'weight': weight,
            'regime_weights': {
                'BULL': 1.0, 'BEAR': 1.0, 'SIDEWAYS': 1.0, 'TURBULENT': 0.5
            }
        })
        self.performance_history[name] = []

    def get_signal(
        self,
        features: pd.DataFrame,
        regime: str = "SIDEWAYS",
        regime_probabilities: Optional[Dict[str, float]] = None
    ) -> int:
        """
        Aggregates model signals into a unified decision:
          1  : BUY
         -1  : SELL
          0  : NO_TRADE / HOLD
        """
        if not self.models or features.empty:
            return 0

        total_signal = 0.0
        total_weight = 0.0

        for item in self.models:
            model = item['model']
            name = item['name']
            base_w = item['weight']

            # Adjust weight by historical hit-rate if available
            hist = self.performance_history.get(name, [])
            hit_rate_mult = 1.0
            if len(hist) >= 5:
                accuracy = sum(hist[-20:]) / len(hist[-20:])
                hit_rate_mult = max(0.2, (accuracy - 0.45) * 3.0 + 1.0)

            # Adjust weight by regime
            regime_mult = item['regime_weights'].get(regime, 1.0)
            if regime_probabilities:
                turbulent_prob = regime_probabilities.get('TURBULENT', 0.0)
                regime_mult *= (1.0 - turbulent_prob * 0.5)

            effective_weight = base_w * hit_rate_mult * regime_mult

            try:
                sig = model.predict(features)
                total_signal += sig * effective_weight
                total_weight += effective_weight
            except Exception as e:
                logger.debug("Ensemble model %s evaluation failed: %s", name, e)

        if total_weight <= 1e-6:
            return 0

        aggregated = total_signal / total_weight

        # Explicit NO_TRADE decision gating
        if aggregated > self.confidence_gate:
            return 1
        elif aggregated < -self.confidence_gate:
            return -1
        else:
            return 0  # NO_TRADE / HOLD

    def record_outcome(self, model_name: str, predicted_signal: int, realized_return: float) -> None:
        """Records outcome hit to adaptively adjust model weights over time."""
        if model_name in self.performance_history:
            correct = 1 if (predicted_signal * realized_return) > 0 else 0
            self.performance_history[model_name].append(correct)
