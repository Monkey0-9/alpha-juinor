"""
Factor Orthogonalization - Remove Multicollinearity.

Uses Gram-Schmidt process to create orthogonal factor exposures,
preventing double-counting of correlated signals.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class FactorOrthogonalizer:
    """
    Orthogonalize correlated factors using Gram-Schmidt.

    Why: Many factors are highly correlated (e.g., momentum 12-1 and 6-1).
    Using them directly causes multicollinearity and unstable weights.
    Orthogonalization creates independent signals.
    """

    def __init__(self, min_correlation: float = 0.7):
        """
        Args:
            min_correlation: Minimum correlation to trigger orthogonalization.
        """
        self.min_correlation = min_correlation
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.orthogonal_basis: Optional[np.ndarray] = None

    def fit(self, factor_matrix: pd.DataFrame) -> "FactorOrthogonalizer":
        """
        Fit the orthogonalizer on historical factor data.

        Args:
            factor_matrix: DataFrame with columns as factors, rows as observations
        """
        # Remove NaN
        clean = factor_matrix.dropna()
        if len(clean) < 20:
            logger.warning("Insufficient data for orthogonalization")
            return self

        # Compute correlation matrix
        self.correlation_matrix = clean.corr()

        # Identify highly correlated factor pairs
        correlated_pairs = self._find_correlated_pairs()

        if correlated_pairs:
            logger.info(
                f"Found {len(correlated_pairs)} highly correlated factor pairs"
            )
            for f1, f2, corr in correlated_pairs[:5]:  # Log top 5
                logger.debug(f"  {f1} <-> {f2}: {corr:.3f}")

        # Compute orthogonal basis via QR decomposition
        self.orthogonal_basis = self._compute_orthogonal_basis(clean.values)

        return self

    def transform(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Transform factors to orthogonal space.

        Args:
            factor_matrix: Original factors

        Returns:
            Orthogonalized factors
        """
        if self.orthogonal_basis is None:
            logger.warning("Orthogonalizer not fitted, returning original")
            return factor_matrix

        clean = factor_matrix.dropna()
        if len(clean) == 0:
            return factor_matrix

        # Project onto orthogonal basis
        try:
            # Standardize first
            standardized = (clean - clean.mean()) / clean.std()

            # QR decomposition for orthogonalization
            Q, R = np.linalg.qr(standardized.values)

            # Create orthogonal DataFrame
            ortho_df = pd.DataFrame(
                Q[:, :factor_matrix.shape[1]],
                index=clean.index,
                columns=[f"{c}_ortho" for c in clean.columns]
            )

            return ortho_df

        except Exception as e:
            logger.error(f"Orthogonalization failed: {e}")
            return factor_matrix

    def _find_correlated_pairs(self) -> List[Tuple[str, str, float]]:
        """Find factor pairs with correlation above threshold."""
        if self.correlation_matrix is None:
            return []

        pairs = []
        cols = self.correlation_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = abs(self.correlation_matrix.iloc[i, j])
                if corr >= self.min_correlation:
                    pairs.append((cols[i], cols[j], corr))

        # Sort by correlation (descending)
        return sorted(pairs, key=lambda x: -x[2])

    def _compute_orthogonal_basis(self, X: np.ndarray) -> np.ndarray:
        """Compute orthogonal basis using modified Gram-Schmidt."""
        n, m = X.shape
        Q = np.zeros((n, m))
        R = np.zeros((m, m))

        for j in range(m):
            v = X[:, j].copy()

            for i in range(j):
                R[i, j] = np.dot(Q[:, i], X[:, j])
                v = v - R[i, j] * Q[:, i]

            R[j, j] = np.linalg.norm(v)
            if R[j, j] > 1e-10:
                Q[:, j] = v / R[j, j]
            else:
                Q[:, j] = 0

        return Q

    def get_factor_loadings(self) -> Optional[pd.DataFrame]:
        """Get loadings of original factors on orthogonal components."""
        if self.orthogonal_basis is None:
            return None

        # This would require storing the R matrix from QR
        # For now, return correlation matrix as proxy
        return self.correlation_matrix

    def explain_variance(self, factor_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Compute variance explained by each orthogonal factor.
        """
        ortho = self.transform(factor_matrix)

        total_var = factor_matrix.var().sum()
        ortho_var = ortho.var()

        return {
            col: float(var / total_var)
            for col, var in ortho_var.items()
        }


def orthogonalize_factors(
    factor_df: pd.DataFrame,
    priority_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to orthogonalize factors.

    Factors are orthogonalized in priority order, with earlier
    factors retained more faithfully.

    Args:
        factor_df: DataFrame with factor columns
        priority_order: Order of factors (first = highest priority)

    Returns:
        Orthogonalized factor DataFrame
    """
    if priority_order:
        # Reorder columns by priority
        ordered_cols = [c for c in priority_order if c in factor_df.columns]
        remaining = [c for c in factor_df.columns if c not in ordered_cols]
        factor_df = factor_df[ordered_cols + remaining]

    orthogonalizer = FactorOrthogonalizer()
    orthogonalizer.fit(factor_df)
    return orthogonalizer.transform(factor_df)
