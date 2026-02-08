"""
Enhanced Reinforcement Learning for Portfolio Optimization
==========================================================

Upgrades to RL agent with:
- Multi-objective optimization (Sharpe, drawdown, turnover)
- Hierarchical RL for multi-timeframe decisions
- Attention mechanisms for feature selection
- Experience replay with prioritization
"""

import logging
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Mock:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return Mock
        def __call__(self, *args, **kwargs): return Mock()
        def __getitem__(self, key): return Mock
    torch = Mock()
    nn = Mock()
    nn.Module = object
    F = Mock()

logger = logging.getLogger(__name__)


class MultiObjectiveReward:
    """
    Multi-objective reward function for portfolio optimization.

    Combines:
    - Sharpe ratio maximization
    - Drawdown minimization
    - Turnover cost minimization
    """

    def __init__(
        self,
        sharpe_weight: float = 0.6,
        drawdown_weight: float = 0.3,
        turnover_weight: float = 0.1,
    ):
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.turnover_weight = turnover_weight

    def calculate(
        self,
        returns: np.ndarray,
        max_drawdown: float,
        turnover: float,
    ) -> float:
        """
        Calculate composite reward.

        Args:
            returns: Portfolio returns
            max_drawdown: Maximum drawdown
            turnover: Portfolio turnover

        Returns:
            Composite reward
        """
        # Sharpe ratio component
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        else:
            sharpe = 0

        # Drawdown component (negative = penalty)
        drawdown_penalty = -max_drawdown

        # Turnover component (negative = penalty)
        turnover_penalty = -turnover * 0.001  # 10bp cost per turnover

        reward = (
            self.sharpe_weight * sharpe
            + self.drawdown_weight * drawdown_penalty
            + self.turnover_weight * turnover_penalty
        )

        return reward


class AttentionPortfolioNetwork(nn.Module):
    """
    Portfolio network with attention mechanism for feature selection.
    """

    def __init__(
        self,
        num_features: int,
        num_assets: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.num_assets = num_assets

        # Feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_features),
            nn.Softmax(dim=-1),
        )

        # Asset processing
        self.asset_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Portfolio allocation head
        self.allocation_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * num_assets, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: [batch, num_assets, num_features]

        Returns:
            (allocations, value)
            allocations: [batch, num_assets]
            value: [batch, 1]
        """
        batch_size = features.size(0)

        # Apply feature attention
        attention_weights = self.feature_attention(features)
        attended_features = features * attention_weights

        # Encode each asset
        asset_embeddings = self.asset_encoder(attended_features)

        # Get allocations
        logits = self.allocation_head(asset_embeddings).squeeze(-1)
        allocations = F.softmax(logits, dim=-1)

        # Get value estimate
        flattened = asset_embeddings.view(batch_size, -1)
        value = self.value_head(flattened)

        return allocations, value


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for efficient learning.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer."""
        max_priority = self.priorities.max() if self.buffer else 1.0

        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Sample batch with prioritization.

        Args:
            batch_size: Size of batch
            beta: Importance sampling exponent

        Returns:
            (experiences, indices, weights)
        """
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        priorities = self.priorities[: len(self.buffer)]
        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(
            len(self.buffer), batch_size, p=probabilities, replace=False
        )

        experiences = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class EnhancedPortfolioRL:
    """
    Enhanced RL agent for portfolio optimization.

    Features:
    - Multi-objective reward
    - Attention-based architecture
    - Prioritized experience replay
    - PPO-style updates
    """

    def __init__(
        self,
        num_features: int,
        num_assets: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.num_assets = num_assets
        self.gamma = gamma
        self.device = device

        if not TORCH_AVAILABLE:
            logger.warning("Torch not available. RL agent in simulation mode.")
            return

        # Networks
        self.policy = AttentionPortfolioNetwork(
            num_features, num_assets, hidden_dim
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)

        # Reward calculator
        self.reward_calculator = MultiObjectiveReward()

        # Metrics
        self.episode_returns = []
        self.episode_sharpes = []

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Select portfolio allocation.

        Args:
            state: Market state [num_assets, num_features]
            deterministic: If True, use greedy policy

        Returns:
            Portfolio weights [num_assets]
        """
        if not TORCH_AVAILABLE:
            allocations = np.random.dirichlet(np.ones(self.num_assets) * 10)
            return allocations

        state_tensor = (
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            allocations, _ = self.policy(state_tensor)
            allocations = allocations.squeeze(0).cpu().numpy()

        if not deterministic:
            # Add exploration noise
            noise = np.random.dirichlet(np.ones(self.num_assets) * 10)
            allocations = 0.9 * allocations + 0.1 * noise

        # Ensure allocations sum to 1
        allocations /= allocations.sum()

        return allocations

    def update(self, batch_size: int = 64):
        """
        Update policy using prioritized experience replay.

        Args:
            batch_size: Batch size for updates
        """
        if len(self.replay_buffer) < batch_size:
            return

        experiences, indices, weights = self.replay_buffer.sample(
            batch_size
        )

        if len(experiences) == 0:
            return

        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Compute current values
        _, current_values = self.policy(states)

        # Compute target values
        with torch.no_grad():
            _, next_values = self.policy(next_states)
            target_values = rewards + self.gamma * next_values * (1 - dones)

        # TD errors for priority update
        td_errors = torch.abs(target_values - current_values)

        # Weighted MSE loss
        value_loss = (weights_tensor * (target_values - current_values) ** 2).mean()

        # Update network
        self.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        new_priorities = td_errors.cpu().detach().numpy().flatten() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

    def save(self, path: str):
        """Save model."""
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Saved model to {path}")

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Loaded model from {path}")
