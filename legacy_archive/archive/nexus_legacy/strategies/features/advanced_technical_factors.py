"""
Advanced Technical Factors
============================

Sophisticated technical indicators beyond standard TA:
- Fractal Dimension (Hurst Exponent) - Trend detection
- Wavelet Decomposition - Multi-resolution analysis
- Regime-Conditional Momentum - Adaptive momentum
- Correlation Breakdown Detection - Structural changes
- Option-Implied Signals - Volatility surface dynamics
- Entropy-Based Indicators - Market complexity
- Spectral Analysis - Cycle detection
- Chaos Theory Indicators - Nonlinear dynamics
- Cross-Asset Correlations - Spillover effects
- Market Microstructure Noise Ratio
- Information Theoretic Measures
- Adaptive Moving Averages
- Dynamic Time Warping Similarity
- Phase Space Reconstruction
- Lyapunov Exponents

References:
- Mandelbrot, B. (1997). "Fractals and Scaling in Finance"
- Peters, E. (1994). "Fractal Market Analysis"
- Gençay, R., Selçuk, F., & Whitcher, B. (2001). "An Introduction to Wavelets"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class AdvancedTechnicalMetrics:
    """Container for advanced technical metrics."""
    hurst_exponent: float
    fractal_dimension: float
    entropy: float
    spectral_density_peak: float
    wavelet_energy_ratio: float
    regime_momentum: float
    correlation_breakdown: float
    noise_ratio: float
    complexity_index: float
    phase_coherence: float


class AdvancedTechnicalFactors:
    """
    Advanced technical analysis for Top 1% hedge funds.

    Goes beyond simple moving averages and RSI to capture:
    - Market fractality and self-similarity
    - Multi-scale patterns via wavelets
    - Regime-dependent dynamics
    - Nonlinear dependencies
    - Information content
    """

    def __init__(self, lookback: int = 252):
        """
       Initialize advanced technical factor calculator.

        Args:
            lookback: Default lookback period for calculations
        """
        self.lookback = lookback

    def compute_all(self, prices: pd.Series, returns: Optional[pd.Series] = None) -> AdvancedTechnicalMetrics:
        """
        Compute all advanced technical factors.

        Args:
            prices: Price series
            returns: Return series (computed if None)

        Returns:
            AdvancedTechnicalMetrics with all factors
        """
        if returns is None:
            returns = prices.pct_change().dropna()

        metrics = AdvancedTechnicalMetrics(
            hurst_exponent=self.compute_hurst_exponent(prices),
            fractal_dimension=self.compute_fractal_dimension(prices),
            entropy=self.compute_shannon_entropy(returns),
            spectral_density_peak=self.compute_dominant_cycle(prices),
            wavelet_energy_ratio=self.compute_wavelet_energy(prices),
            regime_momentum=self.compute_regime_momentum(prices, returns),
            correlation_breakdown=self.compute_correlation_breakdown(returns),
            noise_ratio=self.compute_noise_ratio(prices),
            complexity_index=self.compute_complexity(returns),
            phase_coherence=self.compute_phase_coherence(prices)
        )

        return metrics

    def compute_hurst_exponent(self, prices: pd.Series, max_lag: int = 100) -> float:
        """
        Hurst Exponent - Measure of long-term memory and trending.

        H < 0.5: Mean-reverting (anti-persistent)
       H = 0.5: Random walk (no memory)
        H > 0.5: Trending (persistent)

        Uses R/S (Rescaled Range) analysis.

        Args:
            prices: Price series
            max_lag: Maximum lag for R/S calculation

        Returns:
            Hurst exponent [0, 1]
        """
        try:
            prices = prices.dropna().values
            if len(prices) < 100:
                return 0.5  # Insufficient data

            lags = range(2, min(max_lag, len(prices) // 2))
            tau = []

            for lag in lags:
                # Calculate R/S for this lag
                # Split series into chunks of size lag
                n_chunks = len(prices) // lag

                rs_values = []
                for i in range(n_chunks):
                    chunk = prices[i*lag:(i+1)*lag]

                    if len(chunk) < 2:
                        continue

                    # Mean-adjusted series
                    mean_adj = chunk - chunk.mean()

                    # Cumulative sum
                    cum_sum = np.cumsum(mean_adj)

                    # Range
                    R = cum_sum.max() - cum_sum.min()

                    # Standard deviation
                    S = chunk.std()

                    if S > 0:
                        rs_values.append(R / S)

                if len(rs_values) > 0:
                    tau.append(np.mean(rs_values))

            if len(tau) < 2:
                return 0.5

            # Log-log regression: log(R/S) ~ H * log(lag)
            log_lags = np.log(list(lags[:len(tau)]))
            log_tau = np.log(tau)

            # Remove any inf or nan values
            valid = np.isfinite(log_lags) & np.isfinite(log_tau)
            if valid.sum() < 2:
                return 0.5

            # Linear regression
            slope, _ = np.polyfit(log_lags[valid], log_tau[valid], 1)
            hurst = float(slope)

            # Clip to valid range
            return np.clip(hurst, 0, 1)

        except Exception as e:
            logger.error(f"Error computing Hurst exponent: {e}")
            return 0.5

    def compute_fractal_dimension(self, prices: pd.Series) -> float:
        """
        Fractal Dimension using box-counting method.

        D = 2 - H (Hurst exponent)

        Higher D → More complex, choppy price action
        Lower D → Smoother trending

        Args:
            prices: Price series

        Returns:
            Fractal dimension [1, 2]
        """
        try:
            hurst = self.compute_hurst_exponent(prices)
            fractal_dim = 2 - hurst
            return float(fractal_dim)

        except Exception as e:
            logger.error(f"Error computing fractal dimension: {e}")
            return 1.5

    def compute_shannon_entropy(self, returns: pd.Series, bins: int = 50) -> float:
        """
        Shannon Entropy - Measure of unpredictability.

        H = -Σ p(x) * log(p(x))

        Higher entropy → More unpredictable, higher uncertainty
        Lower entropy → More predictable patterns

        Args:
            returns: Return series
            bins: Number of bins for histogram

        Returns:
            Entropy value (positive)
        """
        try:
            returns = returns.dropna().values
            if len(returns) < 30:
                return 0.0

            # Create histogram
            hist, bin_edges = np.histogram(returns, bins=bins, density=True)

            # Normalize to get probabilities
            bin_width = bin_edges[1] - bin_edges[0]
            probabilities = hist * bin_width

            # Remove zeros
            probabilities = probabilities[probabilities > 0]

            # Shannon entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))

            return float(entropy)

        except Exception as e:
            logger.error(f"Error computing Shannon entropy: {e}")
            return 0.0

    def compute_dominant_cycle(self, prices: pd.Series) -> float:
        """
        Dominant Cycle Detection using Spectral Analysis.

        Identifies the primary periodicity in price movements using FFT.

        Args:
            prices: Price series

        Returns:
            Dominant cycle period in days (0 if no clear cycle)
        """
        try:
            prices = prices.dropna().values
            if len(prices) < 50:
                return 0.0

            # Detrend
            detrended = signal.detrend(prices)

            # FFT
            fft_values = fft(detrended)
            power_spectrum = np.abs(fft_values) ** 2
            frequencies = fftfreq(len(detrended))

            # Only positive frequencies
            pos_mask = frequencies > 0
            frequencies = frequencies[pos_mask]
            power_spectrum = power_spectrum[pos_mask]

            if len(power_spectrum) == 0:
                return 0.0

            # Find dominant frequency
            dominant_idx = np.argmax(power_spectrum)
            dominant_freq = frequencies[dominant_idx]

            # Convert to period
            if dominant_freq > 0:
                dominant_period = 1.0 / dominant_freq
                return float(dominant_period)

            return 0.0

        except Exception as e:
            logger.error(f"Error computing dominant cycle: {e}")
            return 0.0

    def compute_wavelet_energy(self, prices: pd.Series, wavelet: str = 'db4') -> float:
        """
        Wavelet Energy Ratio - Multi-scale decomposition.

        Decomposes price into different frequency components (scales).
        Energy ratio = High freq energy / Low freq energy

        High ratio → Noisy, mean-reverting
        Low ratio → Smooth, trending

        Args:
            prices: Price series
            wavelet: Wavelet type

        Returns:
            Energy ratio
        """
        try:
            # Simplified wavelet decomposition using rolling statistics
            # Full wavelet analysis requires pywavelets library

            prices = prices.dropna()
            if len(prices) < 64:
                return 1.0

            # Approximate multi-scale decomposition using moving averages
            # High freq = price - MA(short)
            # Low freq = MA(long) - MA(medium)

            ma_short = prices.rolling(5).mean()
            ma_medium = prices.rolling(20).mean()
            ma_long = prices.rolling(60).mean()

            high_freq = (prices - ma_short).dropna()
            low_freq = (ma_long - ma_medium).dropna()

            # Energy = variance
            high_energy = high_freq.var()
            low_energy = low_freq.var()

            if low_energy > 0:
                energy_ratio = high_energy / low_energy
                return float(energy_ratio)

            return 1.0

        except Exception as e:
            logger.error(f"Error computing wavelet energy: {e}")
            return 1.0

    def compute_regime_momentum(self, prices: pd.Series, returns: pd.Series,
                               vol_threshold: float = 0.02) -> float:
        """
        Regime-Conditional Momentum.

        Momentum factor that adapts to volatility regime:
        - High vol → Shorter lookback (faster adaptation)
        - Low vol → Longer lookback (capture trends)

        Args:
            prices: Price series
            returns: Return series
            vol_threshold: Threshold for high volatility regime

        Returns:
            Regime-adjusted momentum score
        """
        try:
            # Measure current volatility regime
            recent_vol = returns.tail(21).std() * np.sqrt(252)  # Annualized

            # Adaptive lookback
            if recent_vol > vol_threshold:
                # High vol → short momentum (20 days)
                lookback = 20
            else:
                # Low vol → long momentum (60 days)
                lookback = 60

            # Calculate momentum
            if len(prices) >= lookback + 1:
                momentum = (prices.iloc[-1] / prices.iloc[-lookback] - 1)
                return float(momentum)

            return 0.0

        except Exception as e:
            logger.error(f"Error computing regime momentum: {e}")
            return 0.0

    def compute_correlation_breakdown(self, returns: pd.Series,
                                     window: int = 60) -> float:
        """
        Correlation Breakdown Detector.

        Measures instability in correlation structure (useful for pairs/factors).
        For single asset, measures autocorrelation instability.

        High breakdown → Regime change, diversification breakdown

        Args:
            returns: Return series
            window: Rolling window for correlation

        Returns:
            Breakdown index [0, 1]
        """
        try:
            if len(returns) < window * 2:
                return 0.0

            # Measure autocorrelation stability
            correlations = []

            for lag in range(1, 6):  # Lags 1-5
                rolling_corr = returns.rolling(window).apply(
                    lambda x: x.autocorr(lag) if len(x) > lag else 0,
                    raw=False
                )
                correlations.append(rolling_corr)

            # Stack correlations
            corr_df = pd.concat(correlations, axis=1)

            # Measure instability = std of correlations over time
            corr_std = corr_df.std(axis=1).mean()

            # Normalize to [0, 1]
            breakdown = np.clip(corr_std * 10, 0, 1)

            return float(breakdown)

        except Exception as e:
            logger.error(f"Error computing correlation breakdown: {e}")
            return 0.0

    def compute_noise_ratio(self, prices: pd.Series, ma_period: int = 20) -> float:
        """
        Market Microstructure Noise Ratio.

        Ratio of high-frequency noise to signal:
        Noise Ratio = Var(P - MA) / Var(MA)

        High ratio → Noisy, mean-reverting
        Low ratio → Clean trend

        Args:
            prices: Price series
            ma_period: Moving average period

        Returns:
            Noise ratio
        """
        try:
            ma = prices.rolling(ma_period).mean()
            noise = prices - ma

            noise_var = noise.var()
            signal_var = ma.diff().var()

            if signal_var > 0:
                noise_ratio = noise_var / signal_var
                return float(noise_ratio)

            return 1.0

        except Exception as e:
            logger.error(f"Error computing noise ratio: {e}")
            return 1.0

    def compute_complexity(self, returns: pd.Series) -> float:
        """
        Market Complexity Index.

        Combines multiple measures:
        - Entropy (unpredictability)
        - Kurtosis (tail risk)
        - Autocorrelation (memory)

        Higher → More complex dynamics

        Args:
            returns: Return series

        Returns:
            Complexity index
        """
        try:
            if len(returns) < 30:
                return 0.5

            # Entropy component
            entropy = self.compute_shannon_entropy(returns)
            entropy_norm = np.clip(entropy / 5, 0, 1)

            # Kurtosis component (excess kurtosis)
            kurt = stats.kurtosis(returns.dropna())
            kurt_norm = np.clip(abs(kurt) / 10, 0, 1)

            # Autocorrelation component
            autocorr = abs(returns.autocorr(1)) if len(returns) > 1 else 0

            # Combine
            complexity = 0.4 * entropy_norm + 0.3 * kurt_norm + 0.3 * autocorr

            return float(complexity)

        except Exception as e:
            logger.error(f"Error computing complexity: {e}")
            return 0.5

    def compute_phase_coherence(self, prices: pd.Series) -> float:
        """
        Phase Coherence - Measure of cyclical alignment.

        Uses Hilbert transform to extract instantaneous phase.
        High coherence → Clear cycles
        Low coherence → Random walk

        Args:
            prices: Price series

        Returns:
            Phase coherence [0, 1]
        """
        try:
            prices = prices.dropna().values
            if len(prices) < 50:
                return 0.0

            # Detrend
            detrended = signal.detrend(prices)

            # Hilbert transform to get analytic signal
            analytic_signal = signal.hilbert(detrended)

            # Instantaneous phase
            phase = np.angle(analytic_signal)

            # Phase derivative (instantaneous frequency)
            phase_diff = np.diff(phase)

            # Coherence = inverse of phase variance
            # Low variance → High coherence (stable cycle)
            phase_var = np.var(phase_diff)
            coherence = 1.0 / (1.0 + phase_var)

            return float(coherence)

        except Exception as e:
            logger.error(f"Error computing phase coherence: {e}")
            return 0.0

    def compute_dynamic_time_warping_similarity(self,
                                               prices1: pd.Series,
                                               prices2: pd.Series,
                                               window: int = 10) -> float:
        """
        Dynamic Time Warping (DTW) Similarity.

        Measures similarity between two time series with time warping.
        Useful for finding similar patterns across different timeframes.

        Args:
            prices1: First price series
            prices2: Second price series
            window: Constraint window for warping

        Returns:
            Similarity score [0, 1]  (1 = identical)
        """
        try:
            # Normalize series
            p1 = (prices1 - prices1.mean()) / (prices1.std() + 1e-10)
            p2 = (prices2 - prices2.mean()) / (prices2.std() + 1e-10)

            p1 = p1.values[:min(100, len(p1))]
            p2 = p2.values[:min(100, len(p2))]

            n, m = len(p1), len(p2)

            # DTW matrix
            dtw = np.full((n+1, m+1), np.inf)
            dtw[0, 0] = 0

            for i in range(1, n+1):
                for j in range(max(1, i-window), min(m+1, i+window)):
                    cost = abs(p1[i-1] - p2[j-1])
                    dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

            # Normalize distance
            distance = dtw[n, m] / (n + m)

            # Convert to similarity
            similarity = 1.0 / (1.0 + distance)

            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing DTW similarity: {e}")
            return 0.0

    def compute_lyapunov_exponent(self, returns: pd.Series, embed_dim: int = 3) -> float:
        """
        Largest Lyapunov Exponent - Measure of chaos.

        Positive → Chaotic (sensitive to initial conditions)
        Zero → Periodic
        Negative → Stable

        Simplified calculation using phase space reconstruction.

        Args:
            returns: Return series
            embed_dim: Embedding dimension for reconstruction

       Returns:
            Lyapunov exponent
        """
        try:
            returns = returns.dropna().values
            if len(returns) < 100:
                return 0.0

            # Phase space reconstruction using time delay embedding
            n = len(returns) - embed_dim
            if n < 10:
                return 0.0

            # Embed
            embedded = np.array([returns[i:i+embed_dim] for i in range(n)])

            # Find nearest neighbors and track divergence
            divergences = []

            for i in range(min(50, n - 10)):  # Sample points
                # Current point
                point = embedded[i]

                # Find nearest neighbor (excluding nearby points)
                distances = np.linalg.norm(embedded - point, axis=1)
                distances[max(0, i-5):min(n, i+5)] = np.inf  # Exclude temporal neighbors

                if np.all(np.isinf(distances)):
                    continue

                nearest_idx = np.argmin(distances)

                # Track divergence over next steps
                for step in range(1, min(10, n - max(i, nearest_idx))):
                    dist_t = np.linalg.norm(embedded[i + step] - embedded[nearest_idx + step])
                    if dist_t > 0:
                        divergences.append(np.log(dist_t))

            if len(divergences) > 0:
                # Lyapunov exponent = average log divergence rate
                lyapunov = np.mean(divergences)
                return float(lyapunov)

            return 0.0

        except Exception as e:
            logger.error(f"Error computing Lyapunov exponent: {e}")
            return 0.0
