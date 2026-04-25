import numpy as np

class SovereignAttentionEngine:
    """
    Temporal Fusion Transformer (TFT) inspired Attention Engine.
    Uses "Self-Attention" to weigh different historical regimes.
    """
    def calculate_attention_weights(self, price_series, context_vectors):
        if len(price_series) < 60: return np.ones(len(price_series)) / len(price_series)
        
        # Simulated Multi-Head Attention
        # Q * K^T / sqrt(dk)
        query = price_series[-10:]
        keys = price_series[-60:-10]
        
        # Calculate scores (dot product attention)
        scores = np.dot(keys, np.resize(query, len(keys)))
        weights = np.exp(scores - np.max(scores))
        weights /= weights.sum()
        
        return weights

    def get_context_alpha(self, weights):
        # High attention to recent volatility increases alpha sensitivity
        return np.max(weights)
