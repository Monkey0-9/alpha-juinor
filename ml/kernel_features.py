"""
Kernel Methods - Non-linear Feature Transformation.

Based on Renaissance Technologies' kernel trick approach:
- RBF (Gaussian) kernel for local patterns
- Polynomial kernel for interaction features
- Kernel PCA for dimensionality reduction
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KernelFeatures:
    """Kernel-transformed features."""
    original_features: np.ndarray
    kernel_features: np.ndarray
    kernel_type: str
    n_components: int


class KernelFeatureTransformer:
    """
    Transform features using kernel methods to reveal hidden patterns.

    Kernels available:
    - RBF (Gaussian): Captures local similarity
    - Polynomial: Captures feature interactions
    - Sigmoid: Neural network-like transformation
    - Linear: For baseline comparison
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        n_components: int = 50
    ):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_components = n_components

        # Landmark points for Nystrom approximation
        self.landmarks: Optional[np.ndarray] = None
        self.fitted = False

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        RBF (Gaussian) kernel: k(x, y) = exp(-gamma * ||x - y||^2)
        """
        # Compute pairwise squared distances
        X_sq = np.sum(X ** 2, axis=1)
        Y_sq = np.sum(Y ** 2, axis=1)
        distances = X_sq[:, np.newaxis] + Y_sq - 2 * X @ Y.T

        return np.exp(-self.gamma * distances)

    def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Polynomial kernel: k(x, y) = (gamma * x.y + coef0)^degree
        """
        return (self.gamma * X @ Y.T + self.coef0) ** self.degree

    def _sigmoid_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Sigmoid kernel: k(x, y) = tanh(gamma * x.y + coef0)
        """
        return np.tanh(self.gamma * X @ Y.T + self.coef0)

    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Linear kernel: k(x, y) = x.y
        """
        return X @ Y.T

    def compute_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix based on kernel type."""
        kernels = {
            "rbf": self._rbf_kernel,
            "polynomial": self._polynomial_kernel,
            "sigmoid": self._sigmoid_kernel,
            "linear": self._linear_kernel
        }

        kernel_fn = kernels.get(self.kernel_type, self._rbf_kernel)
        return kernel_fn(X, Y)

    def fit(self, X: np.ndarray):
        """
        Fit the transformer using Nystrom approximation.

        Select landmark points for efficient kernel computation.
        """
        n_samples = X.shape[0]
        n_landmarks = min(self.n_components, n_samples)

        # Random sampling for landmarks
        indices = np.random.choice(n_samples, n_landmarks, replace=False)
        self.landmarks = X[indices]

        # Compute kernel matrix between landmarks
        K_mm = self.compute_kernel(self.landmarks, self.landmarks)

        # Compute eigendecomposition for approximation
        eigenvalues, eigenvectors = np.linalg.eigh(K_mm)

        # Filter positive eigenvalues
        positive_mask = eigenvalues > 1e-10
        self.eigenvalues = eigenvalues[positive_mask]
        self.eigenvectors = eigenvectors[:, positive_mask]

        self.fitted = True
        logger.info(
            f"Kernel transformer fitted with {len(self.eigenvalues)} components"
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using Nystrom approximation.
        """
        if not self.fitted:
            self.fit(X)

        # Compute kernel between X and landmarks
        K_nm = self.compute_kernel(X, self.landmarks)

        # Apply Nystrom approximation
        # phi(x) = K_nm @ V @ Lambda^(-1/2)
        sqrt_eigenvalues = np.sqrt(self.eigenvalues)
        transformed = K_nm @ self.eigenvectors / sqrt_eigenvalues

        return transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class KernelPCA:
    """
    Kernel PCA for non-linear dimensionality reduction.

    Useful for:
    - Finding non-linear patterns in market data
    - Reducing noise while preserving structure
    - Creating orthogonal feature space
    """

    def __init__(
        self,
        n_components: int = 10,
        kernel_type: str = "rbf",
        gamma: float = 1.0
    ):
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.gamma = gamma

        self.X_fit: Optional[np.ndarray] = None
        self.alphas: Optional[np.ndarray] = None
        self.lambdas: Optional[np.ndarray] = None

    def _center_kernel(self, K: np.ndarray) -> np.ndarray:
        """Center the kernel matrix."""
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    def fit(self, X: np.ndarray):
        """Fit Kernel PCA."""
        self.X_fit = X
        n_samples = X.shape[0]

        # Compute kernel matrix
        transformer = KernelFeatureTransformer(
            kernel_type=self.kernel_type,
            gamma=self.gamma
        )
        K = transformer.compute_kernel(X, X)

        # Center kernel matrix
        K_centered = self._center_kernel(K)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep top components
        n_components = min(self.n_components, n_samples)
        self.lambdas = eigenvalues[:n_components]
        self.alphas = eigenvectors[:, :n_components]

        # Normalize eigenvectors
        self.alphas = self.alphas / np.sqrt(self.lambdas)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted Kernel PCA."""
        if self.X_fit is None:
            raise ValueError("Kernel PCA not fitted")

        transformer = KernelFeatureTransformer(
            kernel_type=self.kernel_type,
            gamma=self.gamma
        )
        K = transformer.compute_kernel(X, self.X_fit)

        # Center with respect to training data
        n_train = self.X_fit.shape[0]
        K_centered = K - np.mean(K, axis=1, keepdims=True)

        return K_centered @ self.alphas

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)

        transformer = KernelFeatureTransformer(
            kernel_type=self.kernel_type,
            gamma=self.gamma
        )
        K = transformer.compute_kernel(X, X)
        K_centered = self._center_kernel(K)

        return K_centered @ self.alphas


def create_kernel_features(
    data: pd.DataFrame,
    feature_columns: List[str],
    kernel_type: str = "rbf",
    n_components: int = 20
) -> pd.DataFrame:
    """
    Create kernel-transformed features from raw data.

    Args:
        data: DataFrame with features
        feature_columns: Columns to transform
        kernel_type: Type of kernel
        n_components: Number of components

    Returns:
        DataFrame with kernel features
    """
    X = data[feature_columns].values

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_normalized = (X - X_mean) / X_std

    # Apply kernel transformation
    transformer = KernelFeatureTransformer(
        kernel_type=kernel_type,
        n_components=n_components
    )
    X_transformed = transformer.fit_transform(X_normalized)

    # Create DataFrame
    kernel_df = pd.DataFrame(
        X_transformed,
        index=data.index,
        columns=[f"kernel_{i}" for i in range(X_transformed.shape[1])]
    )

    return kernel_df


# Global singleton
_kernel_transformer: Optional[KernelFeatureTransformer] = None


def get_kernel_transformer() -> KernelFeatureTransformer:
    """Get or create global kernel transformer."""
    global _kernel_transformer
    if _kernel_transformer is None:
        _kernel_transformer = KernelFeatureTransformer()
    return _kernel_transformer
