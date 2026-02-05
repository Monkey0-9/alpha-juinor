import numpy as np
import logging
import random

logger = logging.getLogger("RLMetaController")

class RLMetaController:
    """
    Reinforcement Learning Controller for Capital Allocation.
    Replaces static weights with policy-derived actions.
    """
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.state_dim = 5 # e.g. [VIX, MarketTrend, PortfolioVol, CashRatio, DayOfWeek]
        self.action_dim = 3 # e.g. [Aggressive, Neutral, Defensive] (Weights) or Continuous
        self.is_trained = False

        # MOCK POLICY (until trained)
        # 0: Defensive (High Cash), 1: Balanced (60/40), 2: Aggressive (Leverage)
        self.last_action = 1

    def get_state(self, market_data):
        """Construct state vector from market variables."""
        # Placeholder: Randomize or fetch real data
        return np.random.rand(self.state_dim)

    def predict_action(self, state):
        """
        Get action from policy.
        If not trained, return heuristic or random exploration.
        """
        if not self.is_trained:
            # Heuristic: If VIX (state[0]) > 0.8, go Defensive (0)
            if state[0] > 0.8:
                return 0
            else:
                return 1 # Balanced

        # In prod: self.model.predict(state)
        return 1

    def get_allocation_weights(self, strategy_list):
        """
        Convert abstract action index to concrete strategy weights.
        """
        state = self.get_state(None)
        action = self.predict_action(state)

        num_strats = len(strategy_list)
        if num_strats == 0:
            return {}

        weights = {}

        if action == 0: # Defensive
            # 50% Cash (implied by lower total weights), Equities lower
            w = 0.5 / num_strats
            for s in strategy_list: weights[s] = w

        elif action == 2: # Aggressive
            # Full allocation + Leverage
            w = 1.2 / num_strats
            for s in strategy_list: weights[s] = w

        else: # Balanced
            w = 1.0 / num_strats
            for s in strategy_list: weights[s] = w

        logger.info(f"RL Action {action} -> Uniform Weight {w:.2f} per strategy")
        return weights

def get_rl_controller():
    return RLMetaController()
