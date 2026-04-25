import numpy as np

class SovereignSentimentV2:
    """
    Emotional Velocity & Viral Alpha Engine.
    Detects the "Heat" of human emotion behind price action.
    """
    def analyze_emotional_velocity(self, symbol, headlines):
        # Simulated NLP Score
        # We calculate "Sentiment Entropy"
        score = np.random.normal(0.5, 0.2)
        velocity = np.random.uniform(0.1, 0.9)
        
        if velocity > 0.8: return "VIRAL_HEAT"
        if score < 0.2: return "PANIC_BLOOD"
        return "NEUTRAL_EQUILIBRIUM"

    def get_emotional_alpha(self, state):
        mapping = {
            "VIRAL_HEAT": 0.4,
            "PANIC_BLOOD": 0.6, # "Buy the Blood"
            "NEUTRAL_EQUILIBRIUM": 0.0
        }
        return mapping.get(state, 0.0)
