"""
Reinforcement Learning Trading Agent - Deep Q-Network.

Self-learning agent that optimizes trading decisions:
- State: market features, position, P&L
- Actions: BUY, SELL, HOLD
- Reward: risk-adjusted returns (Sharpe-like)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import random

logger = logging.getLogger(__name__)


class TradingAction:
    """Trading actions for RL agent."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class RLState:
    """State representation for RL agent."""
    price_features: np.ndarray  # Technical indicators
    position: float  # Current position (-1 to 1)
    unrealized_pnl: float  # Unrealized P&L
    realized_pnl: float  # Realized P&L
    volatility: float  # Recent volatility

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.concatenate([
            self.price_features,
            [self.position, self.unrealized_pnl, self.realized_pnl, self.volatility]
        ])


@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork:
    """
    Simple neural network for Q-value estimation.

    Pure NumPy implementation (no PyTorch/TensorFlow dependency)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dims: List[int] = [64, 32]
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Initialize weights
        self.weights = []
        self.biases = []

        dims = [state_dim] + hidden_dims + [action_dim]
        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            w = np.random.randn(dims[i], dims[i+1]) * scale
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.maximum(0, x)  # ReLU

        # Output layer (no activation)
        x = x @ self.weights[-1] + self.biases[-1]
        return x

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict Q-values for state."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.forward(state)

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.001
    ):
        """Update weights using gradient descent."""
        # Forward pass
        activations = [states]
        x = states

        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.maximum(0, x)  # ReLU
            activations.append(x)

        # Output layer
        q_values = x @ self.weights[-1] + self.biases[-1]

        # Compute gradients (simplified)
        # Loss = MSE between predicted and target Q-values
        batch_size = states.shape[0]

        # Create target Q-values (only update for taken actions)
        q_targets = q_values.copy()
        for i in range(batch_size):
            q_targets[i, actions[i]] = targets[i]

        # Output layer gradient
        d_output = (q_values - q_targets) / batch_size
        d_weights = activations[-1].T @ d_output
        d_biases = d_output.sum(axis=0)

        self.weights[-1] -= learning_rate * d_weights
        self.biases[-1] -= learning_rate * d_biases

        # Backprop through hidden layers
        d_hidden = d_output @ self.weights[-1].T

        for i in range(len(self.weights) - 2, -1, -1):
            # ReLU derivative
            d_hidden = d_hidden * (activations[i+1] > 0)

            d_weights = activations[i].T @ d_hidden
            d_biases = d_hidden.sum(axis=0)

            self.weights[i] -= learning_rate * d_weights
            self.biases[i] -= learning_rate * d_biases

            if i > 0:
                d_hidden = d_hidden @ self.weights[i].T

    def copy_from(self, other: "QNetwork"):
        """Copy weights from another network."""
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()


class RLTradingAgent:
    """
    Deep Q-Network trading agent.

    Features:
    - Experience replay for stable learning
    - Target network for reduced variance
    - Epsilon-greedy exploration
    - Sharpe-based reward shaping
    """

    def __init__(
        self,
        state_dim: int = 20,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_size: int = 10000,
        target_update_freq: int = 100
    ):
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = QNetwork(state_dim)
        self.target_network = QNetwork(state_dim)
        self.target_network.copy_from(self.q_network)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Training state
        self.steps = 0
        self.episode_rewards: List[float] = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)  # Random action

        q_values = self.q_network.predict(state)
        return int(np.argmax(q_values))

    def compute_reward(
        self,
        action: int,
        returns: float,
        position: float,
        volatility: float
    ) -> float:
        """
        Compute reward with Sharpe-like shaping.

        Encourages:
        - Positive returns when holding position
        - Risk-adjusted performance
        - Avoiding excessive trading
        """
        # Base reward = position * returns
        if action == TradingAction.BUY:
            position_reward = returns * 1.0
        elif action == TradingAction.SELL:
            position_reward = -returns * 1.0
        else:
            position_reward = position * returns

        # Risk adjustment (penalize high volatility positions)
        risk_penalty = -abs(position) * volatility * 0.1

        # Trading cost penalty
        trade_penalty = -0.001 if action != TradingAction.HOLD else 0

        reward = position_reward + risk_penalty + trade_penalty
        return reward

    def train_step(self):
        """Single training step using experience replay."""
        if len(self.buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.buffer.sample(self.batch_size)

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        # Compute targets
        # Q(s, a) = r + gamma * max_a' Q_target(s', a')
        next_q_values = self.target_network.predict(next_states)
        max_next_q = np.max(next_q_values, axis=1)

        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Update Q-network
        self.q_network.update(states, actions, targets, self.learning_rate)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)

        return float(np.mean(np.abs(targets - self.q_network.predict(states)[np.arange(len(actions)), actions])))

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer."""
        self.buffer.push(Experience(state, action, reward, next_state, done))

    def get_trading_signal(
        self,
        state: np.ndarray,
        current_position: float
    ) -> Tuple[str, float]:
        """
        Get trading signal from trained agent.

        Returns: (signal, confidence)
        """
        q_values = self.q_network.predict(state).flatten()
        action = int(np.argmax(q_values))

        # Confidence based on Q-value difference
        q_sorted = np.sort(q_values)[::-1]
        confidence = (q_sorted[0] - q_sorted[1]) / (abs(q_sorted[0]) + 1e-6)
        confidence = float(np.clip(confidence, 0, 1))

        action_map = {
            TradingAction.HOLD: "HOLD",
            TradingAction.BUY: "BUY",
            TradingAction.SELL: "SELL"
        }

        return action_map[action], confidence

    def save_model(self, filepath: str):
        """Save model weights."""
        np.savez(
            filepath,
            weights=[w for w in self.q_network.weights],
            biases=[b for b in self.q_network.biases]
        )

    def load_model(self, filepath: str):
        """Load model weights."""
        data = np.load(filepath)
        for i, w in enumerate(data["weights"]):
            self.q_network.weights[i] = w
        for i, b in enumerate(data["biases"]):
            self.q_network.biases[i] = b
        self.target_network.copy_from(self.q_network)


# Global singleton
_rl_agent: Optional[RLTradingAgent] = None


def get_rl_agent() -> RLTradingAgent:
    """Get or create global RL trading agent."""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = RLTradingAgent()
    return _rl_agent
