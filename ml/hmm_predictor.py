"""
Hidden Markov Model Predictor - Renaissance-style Regime Detection.

Based on academic research and Renaissance Technologies practices:
- N-state HMM for market regimes
- Predict state transitions
- Self-calibrating emission probabilities
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """HMM market states."""
    BULL_STRONG = 0
    BULL_WEAK = 1
    NEUTRAL = 2
    BEAR_WEAK = 3
    BEAR_STRONG = 4
    CRISIS = 5


@dataclass
class HMMPrediction:
    """HMM prediction result."""
    current_state: MarketState
    state_probabilities: Dict[MarketState, float]
    transition_probabilities: Dict[MarketState, float]  # Next state probs
    confidence: float
    expected_return: float
    expected_volatility: float


class HiddenMarkovPredictor:
    """
    N-state Hidden Markov Model for regime prediction.

    Implementation:
    - Forward-backward algorithm for state estimation
    - Baum-Welch for parameter learning
    - Viterbi for most likely state sequence

    States:
    - BULL_STRONG: Strong uptrend, low volatility
    - BULL_WEAK: Mild uptrend, moderate volatility
    - NEUTRAL: Sideways, moderate volatility
    - BEAR_WEAK: Mild downtrend, elevated volatility
    - BEAR_STRONG: Strong downtrend, high volatility
    - CRISIS: Extreme volatility, crashes
    """

    def __init__(
        self,
        n_states: int = 6,
        lookback: int = 252,
        learning_rate: float = 0.1
    ):
        self.n_states = n_states
        self.lookback = lookback
        self.learning_rate = learning_rate

        # Transition matrix (A[i,j] = P(state_j | state_i))
        self.transition_matrix = self._initialize_transition_matrix()

        # Emission parameters (mean, std for each state)
        self.emission_params = self._initialize_emission_params()

        # Initial state distribution
        self.initial_distribution = np.ones(n_states) / n_states

        # State history
        self.state_history: List[int] = []
        self.fitted = False

    def _initialize_transition_matrix(self) -> np.ndarray:
        """Initialize transition matrix with regime persistence."""
        # States tend to persist (diagonal dominance)
        A = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            # 80% probability of staying in same state
            A[i, i] = 0.80

            # 10% probability of transitioning to adjacent states
            if i > 0:
                A[i, i-1] = 0.10
            if i < self.n_states - 1:
                A[i, i+1] = 0.10

            # Normalize
            A[i] = A[i] / A[i].sum()

        return A

    def _initialize_emission_params(self) -> Dict[int, Dict[str, float]]:
        """Initialize emission parameters for each state."""
        # (mean daily return, volatility)
        params = {
            0: {"mean": 0.0015, "std": 0.008},   # BULL_STRONG
            1: {"mean": 0.0005, "std": 0.012},   # BULL_WEAK
            2: {"mean": 0.0000, "std": 0.015},   # NEUTRAL
            3: {"mean": -0.0005, "std": 0.018},  # BEAR_WEAK
            4: {"mean": -0.0015, "std": 0.025},  # BEAR_STRONG
            5: {"mean": -0.0030, "std": 0.040},  # CRISIS
        }
        return params

    def _emission_probability(self, observation: float, state: int) -> float:
        """Calculate emission probability P(observation | state)."""
        params = self.emission_params[state]
        mean = params["mean"]
        std = params["std"]

        # Gaussian emission
        exponent = -0.5 * ((observation - mean) / std) ** 2
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)

    def forward_algorithm(self, observations: np.ndarray) -> np.ndarray:
        """
        Forward algorithm: compute P(state_t | obs_1:t).

        Returns alpha[t, i] = P(obs_1:t, state_t = i)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Initialize
        for i in range(self.n_states):
            alpha[0, i] = (
                self.initial_distribution[i] *
                self._emission_probability(observations[0], i)
            )

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                emission = self._emission_probability(observations[t], j)
                alpha[t, j] = emission * sum(
                    alpha[t-1, i] * self.transition_matrix[i, j]
                    for i in range(self.n_states)
                )

        return alpha

    def backward_algorithm(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward algorithm: compute P(obs_t+1:T | state_t).

        Returns beta[t, i] = P(obs_t+1:T | state_t = i)
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Initialize
        beta[T-1, :] = 1.0

        # Backward pass
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = sum(
                    self.transition_matrix[i, j] *
                    self._emission_probability(observations[t+1], j) *
                    beta[t+1, j]
                    for j in range(self.n_states)
                )

        return beta

    def viterbi(self, observations: np.ndarray) -> List[int]:
        """
        Viterbi algorithm: find most likely state sequence.
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        for i in range(self.n_states):
            delta[0, i] = (
                self.initial_distribution[i] *
                self._emission_probability(observations[0], i)
            )

        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                emission = self._emission_probability(observations[t], j)
                probs = [
                    delta[t-1, i] * self.transition_matrix[i, j]
                    for i in range(self.n_states)
                ]
                psi[t, j] = np.argmax(probs)
                delta[t, j] = emission * probs[psi[t, j]]

        # Backtrack
        states = [0] * T
        states[T-1] = np.argmax(delta[T-1, :])

        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def fit(self, returns: pd.Series, max_iterations: int = 100):
        """
        Fit HMM using Baum-Welch algorithm.
        """
        observations = returns.values[-self.lookback:]
        T = len(observations)

        for iteration in range(max_iterations):
            # E-step: Forward-backward
            alpha = self.forward_algorithm(observations)
            beta = self.backward_algorithm(observations)

            # Compute gamma (state occupancy) and xi (transitions)
            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T-1, self.n_states, self.n_states))

            for t in range(T):
                denominator = sum(
                    alpha[t, i] * beta[t, i]
                    for i in range(self.n_states)
                )
                for i in range(self.n_states):
                    gamma[t, i] = (alpha[t, i] * beta[t, i]) / max(denominator, 1e-10)

            for t in range(T-1):
                denominator = sum(
                    alpha[t, i] * self.transition_matrix[i, j] *
                    self._emission_probability(observations[t+1], j) * beta[t+1, j]
                    for i in range(self.n_states)
                    for j in range(self.n_states)
                )
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (
                            alpha[t, i] * self.transition_matrix[i, j] *
                            self._emission_probability(observations[t+1], j) *
                            beta[t+1, j]
                        ) / max(denominator, 1e-10)

            # M-step: Update parameters
            # Update transition matrix
            for i in range(self.n_states):
                denom = gamma[:-1, i].sum()
                for j in range(self.n_states):
                    if denom > 0:
                        new_val = xi[:, i, j].sum() / denom
                        self.transition_matrix[i, j] = (
                            (1 - self.learning_rate) * self.transition_matrix[i, j] +
                            self.learning_rate * new_val
                        )

            # Update emission parameters
            for i in range(self.n_states):
                denom = gamma[:, i].sum()
                if denom > 0:
                    new_mean = (gamma[:, i] * observations).sum() / denom
                    new_var = (
                        gamma[:, i] * (observations - new_mean) ** 2
                    ).sum() / denom

                    self.emission_params[i]["mean"] = (
                        (1 - self.learning_rate) * self.emission_params[i]["mean"] +
                        self.learning_rate * new_mean
                    )
                    self.emission_params[i]["std"] = (
                        (1 - self.learning_rate) * self.emission_params[i]["std"] +
                        self.learning_rate * max(np.sqrt(new_var), 0.001)
                    )

        self.fitted = True
        logger.info("HMM fitted successfully")

    def predict(self, returns: pd.Series) -> HMMPrediction:
        """
        Predict current state and next state probabilities.
        """
        observations = returns.values[-self.lookback:]

        # Get current state probabilities
        alpha = self.forward_algorithm(observations)
        state_probs = alpha[-1] / alpha[-1].sum()

        # Most likely current state
        current_state_idx = np.argmax(state_probs)
        current_state = MarketState(current_state_idx)

        # Next state probabilities
        next_probs = self.transition_matrix[current_state_idx]

        # Expected return and volatility
        expected_return = sum(
            state_probs[i] * self.emission_params[i]["mean"]
            for i in range(self.n_states)
        )
        expected_volatility = sum(
            state_probs[i] * self.emission_params[i]["std"]
            for i in range(self.n_states)
        )

        # Confidence based on state probability concentration
        confidence = float(max(state_probs))

        return HMMPrediction(
            current_state=current_state,
            state_probabilities={
                MarketState(i): float(state_probs[i])
                for i in range(self.n_states)
            },
            transition_probabilities={
                MarketState(i): float(next_probs[i])
                for i in range(self.n_states)
            },
            confidence=confidence,
            expected_return=float(expected_return),
            expected_volatility=float(expected_volatility)
        )

    def get_trading_signal(self, prediction: HMMPrediction) -> Tuple[str, float]:
        """
        Generate trading signal from HMM prediction.

        Returns: (signal, strength)
        """
        state = prediction.current_state
        confidence = prediction.confidence

        if state in [MarketState.BULL_STRONG, MarketState.BULL_WEAK]:
            return "BUY", confidence
        elif state in [MarketState.BEAR_STRONG, MarketState.BEAR_WEAK, MarketState.CRISIS]:
            return "SELL", confidence
        else:
            return "HOLD", confidence


# Global singleton
_hmm: Optional[HiddenMarkovPredictor] = None


def get_hmm_predictor() -> HiddenMarkovPredictor:
    """Get or create global HMM predictor."""
    global _hmm
    if _hmm is None:
        _hmm = HiddenMarkovPredictor()
    return _hmm
