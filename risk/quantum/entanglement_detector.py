"""
risk/quantum/entanglement_detector.py

Market Entanglement & Phase Transition Detection.
Uses Tail Mutual Information to detect systemic coupling.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List
from .contracts import EntanglementReport

logger = logging.getLogger("ENTANGLEMENT")

class EntanglementDetector:
    """
    Detects non-linear coupling (entanglement) between assets.
    """

    def __init__(self, threshold: float = 0.7, quantile: float = 0.05):
        self.threshold = threshold
        self.q = quantile

    def compute_metric(self, returns: pd.DataFrame) -> EntanglementReport:
        """
        Compute global entanglement index and asset centrality.
        """
        if returns.empty:
            return EntanglementReport(0.0, {}, "empty", False)

        # 1. Tail Indicators (1 if return <= q-quantile, else 0)
        # Using empirical quantile per asset
        thresholds = returns.quantile(self.q)
        indicators = (returns <= thresholds).astype(int)

        # 2. Adjacency Matrix (Jaccard Index or MI proxy)
        # E_ij = P(i & j) / sqrt(P(i)P(j)) — Normalized Co-occurrence
        T = len(returns)
        cooccurrence = indicators.T @ indicators  # (N, N)
        probs = cooccurrence / T

        # Normalize (Correlation-like)
        # Avoid div zero
        with np.errstate(divide='ignore', invalid='ignore'):
             diag = np.sqrt(np.diag(probs))
             outer = np.outer(diag, diag)
             adj = probs / outer
             adj[np.isnan(adj)] = 0.0

        # 3. Eigen Analysis
        try:
            eigvals, eigvecs = np.linalg.eigh(adj)
            # Global Entanglement = Top Eigenvalue / N (Spectral density concentration)
            top_eig = np.max(eigvals)
            global_index = top_eig / len(returns.columns)

            # Asset Centrality = Top Eigenvector^2 (Principal component contribution)
            top_vec = eigvecs[:, -1] # corresponding to max eig
            centrality = (top_vec ** 2) / np.sum(top_vec ** 2)

            asset_scores = {
                col: float(score)
                for col, score in zip(returns.columns, centrality)
            }

        except np.linalg.LinAlgError:
            global_index = 0.0
            asset_scores = {c: 0.0 for c in returns.columns}

        breach = global_index > self.threshold

        return EntanglementReport(
            global_index=float(global_index),
            asset_centrality=asset_scores,
            adjacency_matrix_id=f"adj_{len(returns)}",
            threshold_breach=breach
        )
