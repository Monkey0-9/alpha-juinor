import logging
import numpy as np
from typing import List, Optional, Union
from numba import jit

logger = logging.getLogger(__name__)


class KalmanFilter:
    """State-Space Model for Signal Denoising.

    Estimates the 'True' Market Price by filtering out microstructure
    noise using a recursive Bayesian estimation process.

    Mathematical Basis:
    - Predict Step:
        x_k^- = x_{k-1}  (State prediction)
        P_k^- = P_{k-1} + Q  (Error covariance prediction)
    - Update Step:
        K_k = P_k^- / (P_k^- + R)  (Kalman Gain)
        x_k = x_k^- + K_k * (z_k - x_k^-)  (State update)
        P_k = (1 - K_k) * P_k^-  (Error covariance update)

    Where Q is process variance and R is measurement variance.
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-3,
    ):
        self.post_estimate = 0.0
        self.post_error_covariance = 1.0
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def update(self, measurement: float) -> float:
        """Perform the Predict and Update steps."""
        pri_estimate = self.post_estimate
        pri_error_covariance = (
            self.post_error_covariance + self.process_variance
        )

        gain = pri_error_covariance / (
            pri_error_covariance + self.measurement_variance
        )
        self.post_estimate = (
            pri_estimate + gain * (measurement - pri_estimate)
        )
        self.post_error_covariance = (1 - gain) * pri_error_covariance

        return float(self.post_estimate)

    def batch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to an entire data series."""
        if len(data) == 0:
            return np.array([])
        estimates = []
        self.post_estimate = data[0]

        for val in data:
            estimates.append(self.update(val))

        return np.array(estimates)


class TrendAccelerationModel:
    """Finite-difference trend model for price trajectory estimation.

    Computes 1st-order (velocity) and 2nd-order (acceleration)
    numerical derivatives of the price series and combines them
    into a single directional "flow force" score.

    This is NOT a Neural ODE — it uses simple finite differences
    to estimate the instantaneous drift of the price path.
    """

    def predict_trajectory(self, prices: np.ndarray) -> float:
        if len(prices) < 10:
            return 0.0

        velocity = np.diff(prices)
        acceleration = np.diff(velocity)

        flow_force = (
            np.mean(velocity[-5:])
            + 0.5 * np.mean(acceleration[-3:])
        )
        return float(flow_force)

    def get_drift_velocity(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.0
        return float(np.mean(np.diff(prices[-10:])))


# Backwards-compatible alias
NeuralODE = TrendAccelerationModel


class VolatilityTopologyHeuristic:
    """Statistical topology heuristic for market regime detection.

    Uses variance and entropy-like measures to classify the current
    market data cloud into qualitative shape categories:
      CHAOTIC   — high entropy, unstable price manifold
      EXPANDING — high variance, directional expansion
      STABLE    — low variance, mean-reverting
      FLAT      — insufficient data

    This is NOT a Persistent Homology / TDA Mapper implementation.
    It uses simple statistical proxies for topological complexity.
    """

    def map_market_topology(self, data_points: np.ndarray) -> str:
        if len(data_points) < 50:
            return "FLAT"

        variance = np.var(data_points)
        entropy = -np.sum(
            np.abs(data_points)
            * np.log(np.abs(data_points) + 1e-9)
        )

        if entropy > 1000:
            return "CHAOTIC"
        if variance > 500:
            return "EXPANDING"
        return "STABLE"

    def get_topological_risk(self, topology: str) -> float:
        risk_map = {
            "CHAOTIC": 0.9,
            "EXPANDING": 0.5,
            "STABLE": 0.1,
            "FLAT": 0.3,
        }
        return risk_map.get(topology, 0.5)


# Backwards-compatible alias
TDAMapper = VolatilityTopologyHeuristic


class FractalEngine:
    """Fractal Dimension (FD) of market structure via Katz algorithm.

    FD = log(L) / log(d), where L is total path length and d is
    the diameter.
    - FD ~ 1.0: Linear, highly persistent trend.
    - FD ~ 1.5: Brownian motion (Random Walk).
    - FD ~ 2.0: Space-filling, highly mean-reverting/chaotic.
    """

    @staticmethod
    @jit(nopython=True)
    def _fast_fd(prices: np.ndarray) -> float:
        n = len(prices)
        if n < 30:
            return 1.5

        variation = 0.0
        for i in range(1, n):
            variation += abs(prices[i] - prices[i - 1])

        if variation == 0:
            return 1.0

        fd = 1.0 + (np.log(variation) / np.log(float(n)))
        return min(2.0, max(1.0, fd))

    def calculate_dimension(self, prices: np.ndarray) -> float:
        return self._fast_fd(prices)

    def analyze_structure(self, fd: float) -> str:
        if fd > 1.7:
            return "HIGH_ROUGHNESS"
        if fd < 1.3:
            return "HIGH_PERSISTENCE"
        return "STABLE"
