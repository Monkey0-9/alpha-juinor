import numpy as np
import logging

logger = logging.getLogger("MonteCarloSim")

class SovereignMonteCarlo:
    """
    Simulates 10,000 future paths for a trade to calculate Survival Probability.
    """
    def __init__(self, simulations=10000, horizon=10):
        self.simulations = simulations
        self.horizon = horizon

    def simulate_trade_survival(self, current_price, vol, stop_loss):
        """
        Calculates the probability that the trade does NOT hit the stop loss.
        """
        dt = 1/252
        # Geometric Brownian Motion simulation
        random_shocks = np.random.standard_normal((self.simulations, self.horizon))
        drift = 0.0 # Neutral drift for survival check
        
        # Calculate paths
        paths = np.zeros((self.simulations, self.horizon))
        paths[:, 0] = current_price
        
        for t in range(1, self.horizon):
            paths[:, t] = paths[:, t-1] * np.exp(
                (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * random_shocks[:, t]
            )

        # Survival check
        min_prices = np.min(paths, axis=1)
        survival_count = np.sum(min_prices > stop_loss)
        survival_prob = survival_count / self.simulations
        
        return survival_prob
