import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import random
import json
from pathlib import Path
import asyncio
import threading
import time

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    TREND_FOLLOWER = "trend_follower"
    MEAN_REVERTER = "mean_reverter"
    MOMENTUM_TRADER = "momentum_trader"
    VOLATILITY_TRADER = "volatility_trader"
    SENTIMENT_TRADER = "sentiment_trader"
    ARBITRAGEUR = "arbitrageur"
    MARKET_MAKER = "market_maker"
    RISK_MANAGER = "risk_manager"

class CommunicationProtocol(Enum):
    BROADCAST = "broadcast"      # All agents receive message
    DIRECT = "direct"           # Specific agent receives message
    AUCTION = "auction"         # Competitive bidding mechanism
    CONSENSUS = "consensus"     # Agreement-based decision making

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender_id: str
    receiver_id: Optional[str]
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    ttl: int = 10  # Time to live in communication cycles

@dataclass
class AgentState:
    """Represents the state of an individual agent."""
    agent_id: str
    role: AgentRole
    portfolio_value: float
    positions: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    last_action: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    cooperation_score: float = 0.0

@dataclass
class MultiAgentExperience:
    """Experience tuple for multi-agent learning."""
    global_state: np.ndarray
    agent_states: Dict[str, np.ndarray]
    joint_action: Dict[str, Any]
    reward: float
    agent_rewards: Dict[str, float]
    next_global_state: np.ndarray
    next_agent_states: Dict[str, np.ndarray]
    done: bool

class InstitutionalMultiAgentRL:
    """
    INSTITUTIONAL-GRADE MULTI-AGENT REINFORCEMENT LEARNING SYSTEM
    Cooperative and competitive multi-agent learning for sophisticated trading strategies.
    Implements communication protocols, role specialization, and emergent behavior.
    """

    def __init__(self, num_agents: int = 8, model_dir: str = "models/multi_agent"):
        self.num_agents = num_agents
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Agent management
        self.agents: Dict[str, Any] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.agent_roles: Dict[str, AgentRole] = {}

        # Communication system
        self.message_queue: deque = deque()
        self.communication_history: List[AgentMessage] = []

        # Learning components
        self.global_replay_buffer = deque(maxlen=10000)
        self.agent_replay_buffers: Dict[str, deque] = {}

        # Coordination mechanisms
        self.coordination_protocols = {
            'consensus': self._consensus_protocol,
            'auction': self._auction_protocol,
            'hierarchical': self._hierarchical_protocol
        }

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.agent_contributions: Dict[str, List[float]] = defaultdict(list)

        # Training parameters
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        # Multi-agent specific parameters
        self.communication_frequency = 5  # Communicate every 5 steps
        self.cooperation_bonus = 0.1
        self.competition_penalty = 0.05

        # Initialize agents
        self._initialize_agents()

        logger.info(f"Multi-Agent RL System initialized with {num_agents} agents")

    def _initialize_agents(self):
        """Initialize specialized trading agents."""
        # Define agent role distribution
        role_distribution = {
            AgentRole.TREND_FOLLOWER: 2,
            AgentRole.MEAN_REVERTER: 2,
            AgentRole.MOMENTUM_TRADER: 1,
            AgentRole.VOLATILITY_TRADER: 1,
            AgentRole.SENTIMENT_TRADER: 1,
            AgentRole.RISK_MANAGER: 1
        }

        agent_id = 0
        for role, count in role_distribution.items():
            for i in range(count):
                agent_name = f"{role.value}_{i}"
                self.agent_roles[agent_name] = role

                # Initialize agent state
                self.agent_states[agent_name] = AgentState(
                    agent_id=agent_name,
                    role=role,
                    portfolio_value=100000.0,  # Starting capital
                    positions={},
                    confidence=0.5
                )

                # Initialize agent model (simplified neural network)
                self.agents[agent_name] = self._create_agent_model(role)

                # Initialize replay buffer for agent
                self.agent_replay_buffers[agent_name] = deque(maxlen=5000)

                agent_id += 1

    def _create_agent_model(self, role: AgentRole) -> Dict[str, Any]:
        """Create a specialized model for the agent role."""
        # Simplified model representation - in production would be neural networks
        model_config = {
            'input_dim': 50,  # Market state features
            'hidden_dim': 64,
            'output_dim': 3,  # Buy, Hold, Sell
            'learning_rate': self.learning_rate
        }

        # Role-specific adjustments
        if role == AgentRole.TREND_FOLLOWER:
            model_config['output_dim'] = 5  # More granular trend actions
        elif role == AgentRole.VOLATILITY_TRADER:
            model_config['input_dim'] = 60  # Additional volatility features
        elif role == AgentRole.SENTIMENT_TRADER:
            model_config['input_dim'] = 70  # Include sentiment features

        # Placeholder for actual model - would be neural network
        model = {
            'config': model_config,
            'weights': np.random.randn(model_config['input_dim'], model_config['hidden_dim']),
            'biases': np.random.randn(model_config['hidden_dim']),
            'output_weights': np.random.randn(model_config['hidden_dim'], model_config['output_dim']),
            'output_biases': np.random.randn(model_config['output_dim'])
        }

        return model

    def step(self, global_state: np.ndarray, agent_observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Execute one step of multi-agent decision making.
        Returns joint action and communication.
        """
        try:
            # 1. Individual agent decision making
            individual_actions = {}
            for agent_id, observation in agent_observations.items():
                action = self._select_action(agent_id, observation)
                individual_actions[agent_id] = action

            # 2. Inter-agent communication (if needed)
            if len(self.message_queue) > 0 or random.random() < 0.3:  # 30% chance
                communications = self._process_communications()
            else:
                communications = {}

            # 3. Coordination and consensus building
            joint_action = self._coordinate_actions(individual_actions, communications, global_state)

            # 4. Execute joint action in environment
            # This would interface with the trading environment
            action_result = self._execute_joint_action(joint_action, global_state)

            # 5. Calculate rewards
            global_reward, agent_rewards = self._calculate_multi_agent_rewards(
                joint_action, action_result, global_state
            )

            # 6. Store experience
            self._store_experience(global_state, agent_observations, joint_action,
                                 global_reward, agent_rewards, action_result)

            # 7. Learning update
            if len(self.global_replay_buffer) > 100:
                self._update_agents()

            return {
                'joint_action': joint_action,
                'communications': communications,
                'global_reward': global_reward,
                'agent_rewards': agent_rewards,
                'action_result': action_result
            }

        except Exception as e:
            logger.error(f"Multi-agent step failed: {e}")
            return self._get_fallback_action()

    def _select_action(self, agent_id: str, observation: np.ndarray) -> Dict[str, Any]:
        """Select action for individual agent using epsilon-greedy policy."""
        agent = self.agents[agent_id]
        agent_state = self.agent_states[agent_id]

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random action
            action_type = random.choice(['BUY', 'SELL', 'HOLD'])
            confidence = random.uniform(0.1, 0.5)
        else:
            # Greedy action based on model
            q_values = self._forward_pass(agent, observation)
            action_idx = np.argmax(q_values)

            actions = ['BUY', 'SELL', 'HOLD']
            action_type = actions[action_idx]
            confidence = np.max(q_values) / np.sum(q_values)  # Softmax confidence

        # Role-specific action modification
        action = self._apply_role_specialization(agent_id, action_type, observation, confidence)

        # Update agent state
        agent_state.last_action = action
        agent_state.confidence = confidence

        return action

    def _apply_role_specialization(self, agent_id: str, base_action: str,
                                 observation: np.ndarray, confidence: float) -> Dict[str, Any]:
        """Apply role-specific modifications to actions."""
        role = self.agent_roles[agent_id]

        action = {
            'agent_id': agent_id,
            'action_type': base_action,
            'confidence': confidence,
            'role': role.value,
            'timestamp': datetime.utcnow()
        }

        # Role-specific logic
        if role == AgentRole.TREND_FOLLOWER:
            # Stronger trend-following bias
            trend_strength = observation[10] if len(observation) > 10 else 0
            if trend_strength > 0.7:
                action['action_type'] = 'STRONG_BUY' if base_action == 'BUY' else base_action
            elif trend_strength < -0.7:
                action['action_type'] = 'STRONG_SELL' if base_action == 'SELL' else base_action

        elif role == AgentRole.MEAN_REVERTER:
            # Mean reversion logic
            z_score = observation[15] if len(observation) > 15 else 0
            if abs(z_score) > 2.0:
                action['action_type'] = 'BUY' if z_score < -2.0 else 'SELL'
                action['confidence'] = min(confidence + 0.3, 1.0)

        elif role == AgentRole.VOLATILITY_TRADER:
            # Volatility-based decisions
            volatility = observation[20] if len(observation) > 20 else 0.02
            if volatility > 0.04:  # High volatility
                action['action_type'] = 'HOLD'  # Wait for volatility to decrease
                action['confidence'] = 0.8

        elif role == AgentRole.SENTIMENT_TRADER:
            # Sentiment-based adjustments
            sentiment = observation[25] if len(observation) > 25 else 0
            if sentiment > 0.5 and base_action == 'BUY':
                action['confidence'] = min(confidence + 0.2, 1.0)
            elif sentiment < -0.5 and base_action == 'SELL':
                action['confidence'] = min(confidence + 0.2, 1.0)

        elif role == AgentRole.RISK_MANAGER:
            # Risk management overrides
            portfolio_var = observation[30] if len(observation) > 30 else 0.02
            if portfolio_var > 0.05:  # High risk
                action['action_type'] = 'REDUCE_RISK'
                action['confidence'] = 0.9

        return action

    def _process_communications(self) -> Dict[str, List[AgentMessage]]:
        """Process pending inter-agent communications."""
        communications = defaultdict(list)

        # Process message queue
        while self.message_queue:
            message = self.message_queue.popleft()

            # Check TTL
            if message.ttl <= 0:
                continue

            # Route message
            if message.receiver_id:
                communications[message.receiver_id].append(message)
            else:
                # Broadcast to all agents
                for agent_id in self.agents.keys():
                    communications[agent_id].append(message)

            # Decrement TTL for next cycle
            message.ttl -= 1
            if message.ttl > 0:
                self.message_queue.append(message)

        return dict(communications)

    def _coordinate_actions(self, individual_actions: Dict[str, Any],
                          communications: Dict[str, List[AgentMessage]],
                          global_state: np.ndarray) -> Dict[str, Any]:
        """Coordinate individual actions into joint action using consensus protocol."""
        # Simple consensus: majority voting with confidence weighting
        action_votes = defaultdict(float)

        for agent_id, action in individual_actions.items():
            action_type = action['action_type']
            confidence = action['confidence']

            # Weight by agent role importance
            role_weight = self._get_role_weight(self.agent_roles[agent_id])
            weighted_confidence = confidence * role_weight

            action_votes[action_type] += weighted_confidence

        # Select winning action
        winning_action = max(action_votes.keys(), key=lambda x: action_votes[x])

        # Calculate consensus confidence
        total_weight = sum(action_votes.values())
        consensus_confidence = action_votes[winning_action] / total_weight if total_weight > 0 else 0

        joint_action = {
            'action_type': winning_action,
            'confidence': consensus_confidence,
            'supporting_agents': [
                agent_id for agent_id, action in individual_actions.items()
                if action['action_type'] == winning_action
            ],
            'individual_actions': individual_actions,
            'coordination_method': 'consensus'
        }

        return joint_action

    def _execute_joint_action(self, joint_action: Dict[str, Any], global_state: np.ndarray) -> Dict[str, Any]:
        """Execute joint action in trading environment (simplified)."""
        # This would interface with actual trading environment
        action_type = joint_action['action_type']

        # Simulate execution result
        if action_type in ['BUY', 'STRONG_BUY']:
            pnl = np.random.normal(0.001, 0.005)  # Small positive return
            slippage = abs(np.random.normal(0, 0.0005))
        elif action_type in ['SELL', 'STRONG_SELL']:
            pnl = np.random.normal(-0.001, 0.005)  # Small negative return
            slippage = abs(np.random.normal(0, 0.0005))
        else:  # HOLD
            pnl = np.random.normal(0, 0.002)  # Near zero return
            slippage = 0

        return {
            'pnl': pnl,
            'slippage': slippage,
            'transaction_cost': 0.0005,  # 5 bps
            'market_impact': abs(pnl) * 0.1,
            'execution_time': np.random.uniform(0.1, 2.0)  # seconds
        }

    def _calculate_multi_agent_rewards(self, joint_action: Dict[str, Any],
                                     action_result: Dict[str, Any],
                                     global_state: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Calculate rewards for multi-agent system."""
        base_reward = action_result['pnl'] - action_result['slippage'] - action_result['transaction_cost']

        # Global reward (team performance)
        global_reward = base_reward

        # Individual agent rewards
        agent_rewards = {}
        supporting_agents = joint_action.get('supporting_agents', [])

        for agent_id in self.agents.keys():
            individual_action = joint_action['individual_actions'][agent_id]
            agent_confidence = individual_action['confidence']

            # Base reward proportional to contribution
            agent_reward = base_reward * agent_confidence

            # Cooperation bonus for consensus
            if agent_id in supporting_agents:
                agent_reward += self.cooperation_bonus * len(supporting_agents)

            # Penalty for conflicting actions
            if individual_action['action_type'] != joint_action['action_type']:
                agent_reward -= self.competition_penalty

            # Role-specific reward adjustments
            agent_reward = self._apply_role_reward_modifier(agent_id, agent_reward, action_result)

            agent_rewards[agent_id] = agent_reward

            # Track agent contributions
            self.agent_contributions[agent_id].append(agent_reward)

        return global_reward, agent_rewards

    def _apply_role_reward_modifier(self, agent_id: str, base_reward: float,
                                  action_result: Dict[str, Any]) -> float:
        """Apply role-specific reward modifications."""
        role = self.agent_roles[agent_id]

        if role == AgentRole.RISK_MANAGER:
            # Risk managers get bonus for low volatility outcomes
            volatility_penalty = action_result.get('market_impact', 0) * 10
            return base_reward - volatility_penalty

        elif role == AgentRole.MARKET_MAKER:
            # Market makers profit from providing liquidity
            liquidity_bonus = action_result.get('execution_time', 1.0) * 0.001
            return base_reward + liquidity_bonus

        elif role == AgentRole.SENTIMENT_TRADER:
            # Sentiment traders get bonus for correct directional calls
            directional_accuracy = 1.0 if action_result['pnl'] > 0 else 0.0
            return base_reward * (1.0 + directional_accuracy * 0.2)

        return base_reward

    def _store_experience(self, global_state: np.ndarray, agent_states: Dict[str, np.ndarray],
                         joint_action: Dict[str, Any], global_reward: float,
                         agent_rewards: Dict[str, float], action_result: Dict[str, Any]):
        """Store experience in replay buffers."""
        # Create next states (simplified - would be actual next observations)
        next_global_state = global_state + np.random.normal(0, 0.001, size=global_state.shape)
        next_agent_states = {
            agent_id: state + np.random.normal(0, 0.001, size=state.shape)
            for agent_id, state in agent_states.items()
        }

        experience = MultiAgentExperience(
            global_state=global_state,
            agent_states=agent_states,
            joint_action=joint_action,
            reward=global_reward,
            agent_rewards=agent_rewards,
            next_global_state=next_global_state,
            next_agent_states=next_agent_states,
            done=random.random() < 0.05  # 5% chance of episode end
        )

        # Store in global buffer
        self.global_replay_buffer.append(experience)

        # Store in individual agent buffers
        for agent_id in agent_rewards.keys():
            agent_experience = {
                'state': agent_states[agent_id],
                'action': joint_action['individual_actions'][agent_id],
                'reward': agent_rewards[agent_id],
                'next_state': next_agent_states[agent_id],
                'done': experience.done
            }
            self.agent_replay_buffers[agent_id].append(agent_experience)

    def _update_agents(self):
        """Update agent policies using experience replay."""
        # Sample batch from global buffer
        if len(self.global_replay_buffer) < 32:
            return

        batch = random.sample(self.global_replay_buffer, min(32, len(self.global_replay_buffer)))

        # Update each agent
        for agent_id in self.agents.keys():
            self._update_agent_policy(agent_id, batch)

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _update_agent_policy(self, agent_id: str, batch: List[MultiAgentExperience]):
        """Update individual agent policy."""
        agent = self.agents[agent_id]

        # Simplified policy update (would be proper gradient descent in production)
        learning_rate = agent['config']['learning_rate']

        for experience in batch:
            state = experience.agent_states[agent_id]
            action = experience.joint_action['individual_actions'][agent_id]
            reward = experience.agent_rewards[agent_id]
            next_state = experience.next_agent_states[agent_id]

            # Simple Q-learning style update
            current_q = self._forward_pass(agent, state)
            next_q = self._forward_pass(agent, next_state)

            # Action index
            action_map = {'BUY': 0, 'HOLD': 1, 'SELL': 2}
            action_idx = action_map.get(action['action_type'], 1)

            # Q-learning update
            target = reward + self.discount_factor * np.max(next_q)
            current_q[action_idx] = (1 - learning_rate) * current_q[action_idx] + learning_rate * target

            # Update weights (simplified)
            # In production, this would be proper backpropagation

    def _forward_pass(self, agent: Dict[str, Any], state: np.ndarray) -> np.ndarray:
        """Forward pass through agent neural network (simplified)."""
        # Simplified neural network forward pass
        hidden = np.tanh(np.dot(state, agent['weights']) + agent['biases'])
        output = np.dot(hidden, agent['output_weights']) + agent['output_biases']

        return output

    def _get_role_weight(self, role: AgentRole) -> float:
        """Get importance weight for agent role in consensus."""
        weights = {
            AgentRole.RISK_MANAGER: 1.5,      # Higher weight for risk management
            AgentRole.TREND_FOLLOWER: 1.2,    # Important for directional moves
            AgentRole.SENTIMENT_TRADER: 1.1,  # Valuable alternative signal
            AgentRole.MEAN_REVERTER: 1.0,     # Standard weight
            AgentRole.MOMENTUM_TRADER: 1.0,   # Standard weight
            AgentRole.VOLATILITY_TRADER: 0.9, # Slightly less weight
            AgentRole.MARKET_MAKER: 0.8,      # Lower weight for specialized role
            AgentRole.ARBITRAGEUR: 0.8        # Lower weight for specialized role
        }

        return weights.get(role, 1.0)

    def _consensus_protocol(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Consensus-based coordination protocol."""
        # Implementation of consensus protocol
        return self._coordinate_actions(actions, {}, np.array([]))

    def _auction_protocol(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Auction-based coordination protocol."""
        # Competitive bidding for action selection
        bids = {}
        for agent_id, action in actions.items():
            bid_amount = action['confidence'] * self._get_role_weight(self.agent_roles[agent_id])
            bids[agent_id] = bid_amount

        # Winner takes all
        winner = max(bids.keys(), key=lambda x: bids[x])
        winning_action = actions[winner]

        return {
            'action_type': winning_action['action_type'],
            'confidence': winning_action['confidence'],
            'winner': winner,
            'coordination_method': 'auction'
        }

    def _hierarchical_protocol(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Hierarchical coordination protocol."""
        # Risk manager has veto power
        risk_manager_action = None
        for agent_id, action in actions.items():
            if self.agent_roles[agent_id] == AgentRole.RISK_MANAGER:
                risk_manager_action = action
                break

        if risk_manager_action and risk_manager_action['confidence'] > 0.8:
            return {
                'action_type': risk_manager_action['action_type'],
                'confidence': risk_manager_action['confidence'],
                'coordination_method': 'hierarchical'
            }

        # Fall back to consensus
        return self._consensus_protocol(actions)

    def _get_fallback_action(self) -> Dict[str, Any]:
        """Get fallback action when step fails."""
        return {
            'joint_action': {'action_type': 'HOLD', 'confidence': 0.1},
            'communications': {},
            'global_reward': 0.0,
            'agent_rewards': {agent_id: 0.0 for agent_id in self.agents.keys()},
            'action_result': {'pnl': 0.0, 'slippage': 0.0, 'transaction_cost': 0.0}
        }

    def save_agents(self, checkpoint_path: str):
        """Save agent states and models."""
        try:
            checkpoint = {
                'agents': self.agents,
                'agent_states': {k: asdict(v) for k, v in self.agent_states.items()},
                'agent_roles': {k: v.value for k, v in self.agent_roles.items()},
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards,
                'agent_contributions': dict(self.agent_contributions),
                'timestamp': datetime.utcnow().isoformat()
            }

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)

            logger.info(f"Multi-agent system saved to {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save agents: {e}")

    def load_agents(self, checkpoint_path: str):
        """Load agent states and models."""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            self.agents = checkpoint['agents']
            self.agent_states = {
                k: AgentState(**{**v, 'role': AgentRole(v['role'])}) if isinstance(v, dict) else v
                for k, v in checkpoint['agent_states'].items()
            }
            self.agent_roles = {k: AgentRole(v) for k, v in checkpoint['agent_roles'].items()}
            self.epsilon = checkpoint['epsilon']
            self.episode_rewards = checkpoint['episode_rewards']
            self.agent_contributions = defaultdict(list, checkpoint['agent_contributions'])

            logger.info(f"Multi-agent system loaded from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load agents: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'num_agents': len(self.agents),
            'agent_roles': {k: v.value for k, v in self.agent_roles.items()},
            'epsilon': self.epsilon,
            'global_buffer_size': len(self.global_replay_buffer),
            'agent_buffer_sizes': {k: len(v) for k, v in self.agent_replay_buffers.items()},
            'total_episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'agent_performance': {
                agent_id: {
                    'avg_contribution': np.mean(contributions) if contributions else 0,
                    'total_contributions': len(contributions),
                    'current_confidence': self.agent_states[agent_id].confidence
                }
                for agent_id, contributions in self.agent_contributions.items()
            }
        }
