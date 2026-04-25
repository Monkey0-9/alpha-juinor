import numpy as np

class SovereignBayesianUpdater:
    """
    Recursive Bayesian Belief Engine.
    Updates the "Probability of Alpha Success" with every new trade and tick.
    """
    def __init__(self, prior_p=0.5):
        self.belief_p = prior_p

    def update_belief(self, signal_strength, actual_outcome):
        """
        Uses Bayes' Theorem: P(A|B) = [P(B|A) * P(A)] / P(B)
        A = Alpha is valid
        B = Correct prediction
        """
        # Likelihood of seeing outcome given alpha is valid
        likelihood = 0.7 if actual_outcome > 0 else 0.3
        
        # Marginal likelihood (Normalization)
        p_outcome = (likelihood * self.belief_p) + (0.5 * (1 - self.belief_p))
        
        # Posterior update
        self.belief_p = (likelihood * self.belief_p) / p_outcome
        return self.belief_p

    def get_confidence_score(self):
        return self.belief_p
