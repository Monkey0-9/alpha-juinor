"""
alpha_families/normalization.py

MANDATORY ALPHA NORMALIZATION LAYER.
Enforces statistical rigor on all alpha signals before they reach PM Brain.

Pipeline:
1. Raw Signal -> Rolling Z-Score (60d window)
2. Clip Z-Score -> [-5, +5]
3. Confidence -> Sigmoid(|Z|) * Data Quality
4. Target Return (mu) -> Z * Volatility * Scale
5. Distribution -> Safe construction with repair logic
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger("ALPHA_NORM")

class AlphaNormalizer:
    def __init__(self,
                 window: int = 60,
                 target_scale: float = 0.5,
                 max_z: float = 5.0):
        """
        Args:
            window: Rolling window for Z-score normalization
            target_scale: Scaling factor for return target (fraction of volatility)
            max_z: Clipping threshold for Z-scores
        """
        self.window = window
        self.target_scale = target_scale
        self.max_z = max_z

    def normalize_signal(self,
                       raw_value: float,
                       history: pd.Series,
                       data_confidence: float = 1.0) -> Tuple[float, float]:
        """
        Normalize a raw signal value using historical distribution.

        Returns:
            (z_score, confidence_score)
        """
        if history.empty or len(history) < 10:
            return 0.0, 0.0 # Insufficient history

        # 1. Compute rolling stats
        # Use simple rolling mean/std from history
        # (Assuming history includes current raw_value or is prior)
        mu = history.mean()
        sigma = history.std()

        if sigma == 0:
            return 0.0, 0.0

        # 2. Compute Z-Score
        z = (raw_value - mu) / sigma

        # 3. Clip Z-Score (Outlier protection)
        z_clipped = max(-self.max_z, min(self.max_z, z))

        # 4. Compute Confidence
        # Higher absolute signal = higher conviction (up to a point)
        # Scaled by data quality
        # Sigmoid: 1 / (1 + exp(-|z|)) -> ranges 0.5 to 1.0 for positive input
        # We map |z| 0..5 to confidence 0..1 roughly
        # Let's use simple tanh for now: tanh(|z|/2) -> 0 at 0, 0.98 at 5
        base_conf = np.tanh(abs(z_clipped) / 2.0)
        final_conf = base_conf * data_confidence

        # Repair bounds
        final_conf = max(0.0, min(1.0, final_conf))

        return z_clipped, final_conf

    def construct_distribution(self,
                             z_score: float,
                             confidence: float,
                             volatility: float) -> Dict[str, float]:
        """
        Construct the AlphaDistribution payload.

        Args:
            z_score: Normalized signal strength (signed)
            confidence: Signal reliability [0, 1]
            volatility: Annualized asset volatility

        Returns:
            Dict matching AlphaDistribution contract
        """
        # Mu (Expected Return) = Direction * Volatility * Scale
        # Stronger signal -> target higher capture of volatility
        # But limited by Z-score clipping
        mu = z_score * volatility * self.target_scale

        # Sigma (Uncertainty)
        # Higher confidence -> Lower uncertainty around mu
        # Base sigma is asset volatility
        # If confidnece is 1.0, sigma = volatility * 0.5 (tight)
        # If confidence is 0.0, sigma = volatility * 2.0 (wide)
        sigma_dist = volatility * (2.0 - 1.5 * confidence)

        # P(Loss) = Probability mu < 0 (for long) or mu > 0 (for short)
        # Simple normal approximation
        # Z_dist = mu / sigma_dist
        # p_loss = 1 - CDF(|Z_dist|)
        # or just empirical mapping

        from scipy.stats import norm
        try:
            if sigma_dist > 0:
                if z_score > 0:
                    p_loss = norm.cdf(0, loc=mu, scale=sigma_dist)
                elif z_score < 0:
                    p_loss = 1.0 - norm.cdf(0, loc=mu, scale=sigma_dist)
                else:
                    p_loss = 0.5
            else:
                p_loss = 0.5
        except:
            p_loss = 0.5

        return {
            "mu": float(mu),
            "sigma": float(sigma_dist),
            "p_loss": float(p_loss),
            "cvar_95": float(mu - 1.645 * sigma_dist), # Parametric Approx
            "confidence": float(confidence)
        }

    def repair_distribution(self, dist: Dict[str, float], price: float = 1.0) -> Dict[str, float]:
        """
        Repair a broken distribution dict with strict institutional normalization.

        Args:
            dist: The distribution dictionary to repair
            price: Current asset price (used for heuristic normalization of price-delta returns)
        """
        cleaned = dist.copy()

        # Extract
        mu = cleaned.get('mu', 0.0)
        sigma = cleaned.get('sigma', 0.0)
        cvar = cleaned.get('cvar_95', -0.01)

        # Handle nan/inf immediately
        if not np.isfinite(mu): mu = 0.0
        if not np.isfinite(sigma): sigma = 0.1
        if not np.isfinite(cvar): cvar = -0.05

        # 1. Heuristic Normalization: Detect Price Deltas vs Returns
        # If mu is massive (e.g. > 1.0 or < -1.0), it might be a price delta
        # User Logic: mu = mu / price if abs(mu) > 1 else mu
        if abs(mu) > 1.0:
            if price > 0:
                mu = mu / price
            else:
                mu = 0.0 # Safety fallback

        # 2. Hard Clipping (Governance Layer)
        # Mu: [-0.05, 0.05] (5% daily alpha is extreme)
        mu = max(-0.05, min(0.05, mu))

        # Sigma: [0.001, 0.10] (0.1% to 10% daily vol)
        sigma = max(0.001, min(0.10, sigma))

        # CVaR: [-0.20, -0.001] (Max 20% daily loss, min 0.1% loss)
        # Note: CVaR is negative
        cvar = max(-0.20, min(-0.001, cvar))

        # 3. Consistency Enforcements
        # Sigma must be positive (handled by clip above)
        # CVaR should be < mu (usually)
        if cvar > mu:
            cvar = mu - (1.645 * sigma)

        cleaned['mu'] = float(mu)
        cleaned['sigma'] = float(sigma)
        cleaned['cvar_95'] = float(cvar)

        # Fix Confidence
        conf = cleaned.get('confidence', 0.0)
        if not np.isfinite(conf): conf = 0.0
        cleaned['confidence'] = max(0.0, min(1.0, conf))

        # Fix P_loss
        pl = cleaned.get('p_loss', 0.5)
        if not np.isfinite(pl): pl = 0.5
        cleaned['p_loss'] = max(0.0, min(1.0, pl))

        return cleaned
