
import numpy as np

class RLAgent:
    """
    Research Only. Never call in live loop.
    Bellman Q Logic.
    """
    def update(self, s, a, r, s_prime, alpha=0.1, gamma=0.99):
        # Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
        pass
    
    def act(self, s):
        return 0
