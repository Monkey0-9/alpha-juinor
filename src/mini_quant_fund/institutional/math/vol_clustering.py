import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VolClustering")

class SovereignHawkesProcess:
    """
    Elite Volatility Clustering Engine.
    Models market 'bursts' using self-exciting process logic.
    Used by top-tier HFT firms to predict 'jump diffusion'.
    """
    def __init__(self, mu=0.01, alpha=0.1, beta=0.5):
        self.mu = mu # Baseline intensity
        self.alpha = alpha # Jump intensity increase
        self.beta = beta # Exponential decay rate

    def estimate_intensity(self, events: np.ndarray, current_time: float) -> float:
        """
        Calculates the conditional intensity lambda(t).
        lambda(t) = mu + sum(alpha * exp(-beta * (t - t_i)))
        """
        if len(events) == 0:
            return self.mu
            
        # Vectorized calculation of self-exciting component
        history_delta = current_time - events[events < current_time]
        intensity = self.mu + np.sum(self.alpha * np.exp(-self.beta * history_delta))
        
        return intensity

    def predict_burst_probability(self, intensity_series: np.ndarray, threshold=0.1) -> bool:
        """
        Predicts if a high-volatility burst is imminent.
        """
        recent_intensity = intensity_series[-1]
        is_burst = recent_intensity > (self.mu + threshold)
        if is_burst:
            logger.warning(f"HIGH INTENSITY DETECTED: {recent_intensity:.4f}. Expect Volatility Cluster.")
        return is_burst

if __name__ == "__main__":
    # Demo: Simulating intensity over a series of market events (timestamps)
    hp = SovereignHawkesProcess()
    timestamps = np.array([1.2, 1.3, 1.35, 2.5, 4.0, 4.1, 4.15, 4.2])
    
    current_intensity = hp.estimate_intensity(timestamps, 4.25)
    print(f"Current Market Intensity: {current_intensity:.4f}")
    hp.predict_burst_probability(np.array([current_intensity]))
