"""
execution_ai/execution_rl.py

Reinforcement Learning for optimal trade execution.
Implements DQN and PPO algorithms for intelligent order execution.
"""
from __future__ import annotations

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None
    F = None
from collections import deque, namedtuple
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None

from monitoring.structured_logger import get_logger

logger = get_logger("execution_rl")

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class RLExecutionConfig:
    """Configuration for RL-based execution."""
    algorithm: str = "DQN"  # DQN or PPO
    state_size: int = 20
    action_size: int = 5
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    episodes: int = 1000
    max_steps: int = 500


if HAS_TORCH:
    class DQNNetwork(nn.Module):
        """Deep Q-Network for execution decisions."""

        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(DQNNetwork, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, action_size)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x


    class PPONetwork(nn.Module):
        """Proximal Policy Optimization network."""

        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(PPONetwork, self).__init__()

            # Shared layers
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)

            # Actor head (policy)
            self.actor = nn.Linear(hidden_size, action_size)

            # Critic head (value)
            self.critic = nn.Linear(hidden_size, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            # Policy and value
            policy_logits = self.actor(x)
            value = self.critic(x)

            return policy_logits, value
else:
    class DQNNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for DQN execution models")


    class PPONetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for PPO execution models")


class ExecutionEnvironment:
    """Custom environment for execution RL training."""

    def __init__(self, market_data: Dict[str, Any], order_params: Dict[str, Any]):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required for RL execution. Install with: pip install gymnasium")

        super(ExecutionEnvironment, self).__init__()

        self.market_data = market_data
        self.order_params = order_params

        # Action space: [hold, aggressive_buy, moderate_buy, moderate_sell, aggressive_sell]
        self.action_space = spaces.Discrete(5)

        # State space: market features + order progress
        self.state_size = 20
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)

        self.reset()

    def reset(self):
        """Reset environment for new episode."""
        self.current_step = 0
        self.remaining_quantity = self.order_params['quantity']
        self.executed_quantity = 0
        self.average_price = 0.0
        self.total_cost = 0.0

        return self._get_state()

    def step(self, action):
        """Execute one step in environment."""
        self.current_step += 1

        # Execute action
        executed_qty, price = self._execute_action(action)

        # Update state
        self.executed_quantity += executed_qty
        self.remaining_quantity -= executed_qty

        if executed_qty > 0:
            self.total_cost += executed_qty * price
            self.average_price = self.total_cost / self.executed_quantity

        # Calculate reward
        reward = self._calculate_reward(action, executed_qty, price)

        # Check if done
        done = (self.remaining_quantity <= 0) or (self.current_step >= self.order_params['max_steps'])

        return self._get_state(), reward, done, {}

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        # Market features
        current_price = self.market_data['prices'][self.current_step] if self.current_step < len(self.market_data['prices']) else self.market_data['prices'][-1]
        volume = self.market_data['volumes'][self.current_step] if self.current_step < len(self.market_data['volumes']) else self.market_data['volumes'][-1]
        volatility = self.market_data['volatility'][self.current_step] if self.current_step < len(self.market_data['volatility']) else self.market_data['volatility'][-1]

        # Order progress
        progress_ratio = self.executed_quantity / self.order_params['quantity'] if self.order_params['quantity'] > 0 else 0
        remaining_ratio = self.remaining_quantity / self.order_params['quantity'] if self.order_params['quantity'] > 0 else 0
        time_ratio = self.current_step / self.order_params['max_steps']

        # Price impact estimation
        price_impact = self._estimate_price_impact()

        # Combine features
        state = np.array([
            current_price / self.order_params['reference_price'] - 1,  # Price relative to reference
            volume / self.market_data['avg_volume'] - 1,  # Volume relative to average
            volatility / self.market_data['avg_volatility'] - 1,  # Volatility relative to average
            progress_ratio,  # Execution progress
            remaining_ratio,  # Remaining quantity ratio
            time_ratio,  # Time progress
            price_impact,  # Estimated price impact
            self.average_price / self.order_params['reference_price'] - 1 if self.average_price > 0 else 0,  # Avg price relative
            self.current_step / 100,  # Normalized step
            len(self.market_data['prices']) - self.current_step,  # Steps remaining
            # Additional features for more complex state
            np.mean(self.market_data['prices'][-10:] if self.current_step >= 10 else self.market_data['prices'][:self.current_step+1]) / current_price - 1,
            np.std(self.market_data['prices'][-10:] if self.current_step >= 10 else self.market_data['prices'][:self.current_step+1]) / current_price,
            self.order_params['urgency'],  # Order urgency
            self.order_params['max_participation_rate'],  # Max participation rate
            self.order_params['side'],  # Order side (buy/sell)
            self.order_params['time_horizon'],  # Time horizon
            price_impact * self.remaining_quantity,  # Total potential impact
            (self.order_params['target_price'] - current_price) / current_price if 'target_price' in self.order_params else 0,  # Distance to target
        ], dtype=np.float32)

        # Pad or truncate to fixed size
        if len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)), 'constant')
        elif len(state) > self.state_size:
            state = state[:self.state_size]

        return state

    def _execute_action(self, action: int) -> Tuple[float, float]:
        """Execute action and return executed quantity and price."""
        current_price = self.market_data['prices'][self.current_step] if self.current_step < len(self.market_data['prices']) else self.market_data['prices'][-1]
        current_volume = self.market_data['volumes'][self.current_step] if self.current_step < len(self.market_data['volumes']) else self.market_data['volumes'][-1]

        # Action mapping
        if action == 0:  # Hold
            return 0.0, current_price
        elif action == 1:  # Aggressive buy
            participation_rate = min(0.2, self.order_params['max_participation_rate'])
        elif action == 2:  # Moderate buy
            participation_rate = min(0.1, self.order_params['max_participation_rate'])
        elif action == 3:  # Moderate sell
            participation_rate = min(0.1, self.order_params['max_participation_rate'])
        elif action == 4:  # Aggressive sell
            participation_rate = min(0.2, self.order_params['max_participation_rate'])
        else:
            return 0.0, current_price

        # Calculate execution quantity
        max_executable = current_volume * participation_rate
        executable_qty = min(max_executable, self.remaining_quantity)

        # Apply price impact
        price_impact = self._calculate_price_impact(executable_qty, current_volume)
        execution_price = current_price * (1 + price_impact) if self.order_params['side'] == 'buy' else current_price * (1 - price_impact)

        return executable_qty, execution_price

    def _calculate_price_impact(self, quantity: float, volume: float) -> float:
        """Calculate price impact based on quantity and volume."""
        # Square-root price impact model
        impact = 0.001 * np.sqrt(quantity / volume) if volume > 0 else 0.01
        return impact

    def _estimate_price_impact(self) -> float:
        """Estimate current price impact."""
        current_volume = self.market_data['volumes'][self.current_step] if self.current_step < len(self.market_data['volumes']) else self.market_data['volumes'][-1]
        return self._calculate_price_impact(self.remaining_quantity, current_volume)

    def _calculate_reward(self, action: int, executed_qty: float, price: float) -> float:
        """Calculate reward for the action."""
        # Base reward for execution
        if executed_qty > 0:
            # Reward for making progress
            progress_reward = executed_qty / self.order_params['quantity']

            # Penalty for price impact
            reference_price = self.order_params['reference_price']
            price_deviation = abs(price - reference_price) / reference_price
            impact_penalty = -price_deviation * 10

            # Time penalty (encourage timely execution)
            time_penalty = -self.current_step / self.order_params['max_steps'] * 0.1

            # Completion bonus
            completion_bonus = 1.0 if self.remaining_quantity <= 0 else 0.0

            return progress_reward + impact_penalty + time_penalty + completion_bonus
        else:
            # Small penalty for holding when there's opportunity
            return -0.001


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for execution."""

    def __init__(self, config: RLExecutionConfig):
        if not HAS_TORCH:
            raise ImportError("torch is required for DQN execution agent")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = DQNNetwork(config.state_size, config.action_size).to(self.device)
        self.target_network = DQNNetwork(config.state_size, config.action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(config.memory_size)

        # Training parameters
        self.epsilon = config.epsilon
        self.steps_done = 0

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                       next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)

    def train(self):
        """Train the network."""
        if len(self.memory) < self.config.batch_size:
            return

        # Sample batch
        experiences = self.memory.sample(self.config.batch_size)
        batch = Experience(*zip(*experiences))

        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)

        # Compute Q-values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.config.gamma * next_q_values * ~done_batch)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

        # Update target network
        self.steps_done += 1
        if self.steps_done % self.config.target_update_freq == 0:
            self.update_target_network()

    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': self.config
        }, filepath)

    def load_model(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']


class PPOAgent:
    """Proximal Policy Optimization agent for execution."""

    def __init__(self, config: RLExecutionConfig):
        if not HAS_TORCH:
            raise ImportError("torch is required for PPO execution agent")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network
        self.network = PPONetwork(config.state_size, config.action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

        # PPO parameters
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)

            # Sample action from policy
            probs = F.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if training:
                action = dist.sample()
            else:
                action = probs.argmax()

            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_trajectory(self, state: np.ndarray, action: int, reward: float,
                       value: float, log_prob: float, done: bool):
        """Store trajectory step."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages."""
        returns = []
        advantages = []

        R = 0
        for i in reversed(range(len(self.rewards))):
            R = self.rewards[i] + self.config.gamma * R * (1 - self.dones[i])
            returns.insert(0, R)

            # Advantage = R - V(s)
            advantage = returns[0] - self.values[i]
            advantages.insert(0, advantage)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def train(self):
        """Train PPO agent."""
        if len(self.states) < self.config.batch_size:
            return

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()

        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)

        # PPO update
        for _ in range(10):  # PPO epochs
            # Forward pass
            policy_logits, values = self.network(states)
            probs = F.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)

            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy loss
            entropy_loss = -dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Clear trajectory
        self.clear_trajectory()

    def clear_trajectory(self):
        """Clear stored trajectory."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)

    def load_model(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ExecutionRLTrainer:
    """Trainer for execution RL agents."""

    def __init__(self, config: RLExecutionConfig):
        self.config = config
        self.logger = logger

        # Initialize agent
        if config.algorithm.upper() == "DQN":
            self.agent = DQNAgent(config)
        elif config.algorithm.upper() == "PPO":
            self.agent = PPOAgent(config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    def train_agent(self, market_data: List[Dict[str, Any]],
                   order_params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the RL agent."""
        self.logger.info(f"Starting {self.config.algorithm} training for {self.config.episodes} episodes")

        training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'execution_costs': [],
            'completion_rates': []
        }

        for episode in range(self.config.episodes):
            # Random order for this episode
            order_params = random.choice(order_params_list)

            # Create environment
            env = ExecutionEnvironment(market_data[episode % len(market_data)], order_params)

            # Train episode
            episode_reward, episode_length, execution_cost, completion_rate = self._train_episode(env)

            # Store metrics
            training_metrics['episode_rewards'].append(episode_reward)
            training_metrics['episode_lengths'].append(episode_length)
            training_metrics['execution_costs'].append(execution_cost)
            training_metrics['completion_rates'].append(completion_rate)

            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(training_metrics['episode_rewards'][-100:])
                avg_cost = np.mean(training_metrics['execution_costs'][-100:])
                avg_completion = np.mean(training_metrics['completion_rates'][-100:])

                self.logger.info(
                    f"Episode {episode}",
                    avg_reward=avg_reward,
                    avg_cost=avg_cost,
                    avg_completion=avg_completion,
                    epsilon=getattr(self.agent, 'epsilon', 'N/A')
                )

        # Save trained model
        model_path = f"models/execution_rl_{self.config.algorithm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        self.agent.save_model(model_path)

        self.logger.info(f"Training completed. Model saved to {model_path}")

        return training_metrics

    def _train_episode(self, env: ExecutionEnvironment) -> Tuple[float, int, float, float]:
        """Train a single episode."""
        state = env.reset()
        total_reward = 0
        steps = 0

        for step in range(self.config.max_steps):
            # Select action
            if isinstance(self.agent, DQNAgent):
                action = self.agent.select_action(state, training=True)
            else:  # PPO
                action, log_prob, value = self.agent.select_action(state, training=True)

            # Execute action
            next_state, reward, done, _ = env.step(action)

            # Store experience
            if isinstance(self.agent, DQNAgent):
                self.agent.store_experience(state, action, reward, next_state, done)
                self.agent.train()
            else:  # PPO
                self.agent.store_trajectory(state, action, reward, value, log_prob, done)
                if done or step == self.config.max_steps - 1:
                    self.agent.train()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Calculate execution metrics
        execution_cost = abs(env.average_price - env.order_params['reference_price']) / env.order_params['reference_price'] if env.average_price > 0 else 1.0
        completion_rate = env.executed_quantity / env.order_params['quantity'] if env.order_params['quantity'] > 0 else 0.0

        return total_reward, steps, execution_cost, completion_rate


# Global trainer instance
execution_rl_trainer = None

def get_execution_rl_trainer(config: RLExecutionConfig = None) -> ExecutionRLTrainer:
    """Get or create execution RL trainer."""
    global execution_rl_trainer
    if execution_rl_trainer is None:
        config = config or RLExecutionConfig()
        execution_rl_trainer = ExecutionRLTrainer(config)
    return execution_rl_trainer
