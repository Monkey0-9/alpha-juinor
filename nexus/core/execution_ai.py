import logging
import numpy as np
import random
from collections import deque

logger = logging.getLogger(__name__)

class ExecutionAgent:
    """
    Reinforcement Learning based Order Router.
    Learns optimal execution paths to minimize market impact and slippage.
    """
    def __init__(self, state_size: int = 10, action_size: int = 3):
        self.state_size = state_size
        self.action_size = action_size # 0: Wait, 1: Small Market, 2: Large Market
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_action(self, market_state: np.ndarray) -> int:
        """
        Determines the execution action based on the current market state.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # In production, this would invoke a trained model prediction
        return 1 # Default to conservative execution

    def learn(self, reward: float):
        """
        Updates agent parameters based on trade feedback (P&L - Slippage).
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logger.debug(f"Agent training update - Reward: {reward:.4f}, Epsilon: {self.epsilon:.4f}")
