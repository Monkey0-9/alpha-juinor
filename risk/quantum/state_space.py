"""
risk/quantum/state_space.py

Regime State Evolution: p(t+1) = U(t)p(t)
Maintains probabilistic belief over K regime states.
"""

import numpy as np
import logging
from typing import List, Optional
from .contracts import QuantumState

logger = logging.getLogger("QUANTUM_STATE")

class RegimeStateSpace:
    """
    Maintains and evolves the regime belief vector.
    """

    def __init__(self, n_regimes: int = 5, seed: int = 42):
        self.K = n_regimes
        self.belief = np.ones(n_regimes) / n_regimes  # Uniform prior
        self.transition_matrix = np.eye(n_regimes)    # Identity default
        np.random.seed(seed)

        # Pre-defined mapping (placeholder for learned model)
        self.regime_labels = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE", "CRASH"]

    def update(self, market_data: Optional[dict] = None) -> QuantumState:
        """
        Evolve state based on market observables.
        p(t+1) = U(t)p(t)
        """
        # In a real implementation, U(t) comes from HMM or RNN
        # For now, we use a slight mean-reversion perturbation

        # Simple toy evolution for now
        perturbation = np.random.normal(0, 0.01, self.K)
        exposed_belief = self.belief + perturbation

        # Softmax to ensure probability constraint
        exp_b = np.exp(exposed_belief)
        self.belief = exp_b / np.sum(exp_b)

        # Calculate entropy
        entropy = -np.sum(self.belief * np.log(self.belief + 1e-9))

        return QuantumState(
            regime_belief=self.belief.tolist(),
            entropy=float(entropy),
            transition_matrix_id="identity_v1"
        )

    def get_compatibility(self, strategy_profile: List[float]) -> float:
        """
        Compute compatibility score r_i = p(t) . C_i
        """
        if len(strategy_profile) != self.K:
            return 0.0
        return float(np.dot(self.belief, strategy_profile))
