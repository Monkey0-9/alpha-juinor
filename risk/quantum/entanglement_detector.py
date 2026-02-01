"""
risk/quantum/entanglement_detector.py

Market Entanglement & Phase Transition Detection.
Uses Tail Mutual Information to detect systemic coupling.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict
from .contracts import EntanglementReport

logger = logging.getLogger("ENTANGLEMENT")


def build_entanglement_matrix(
    returns: pd.DataFrame,
    q: float = 0.05
) -> np.ndarray:
    """
    Construct adjacency matrix based on tail co-occurrence.
    """
    if returns.empty:
        return np.zeros((0, 0))

    # 1. Tail Indicators
    thresholds = returns.quantile(q)
    indicators = (returns <= thresholds).astype(int)

    # 2. Adjacency Matrix
    T = len(returns)
    cooccurrence = indicators.T @ indicators  # (N, N)
    probs = cooccurrence / T

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        diag = np.sqrt(np.diag(probs))
        outer = np.outer(diag, diag)
        adj = probs / outer
        adj[np.isnan(adj)] = 0.0

    return adj.values if isinstance(adj, pd.DataFrame) else adj


def entanglement_indices(
    adj_matrix: np.ndarray
) -> tuple[Dict[str, float], float]:
    """
    Compute spectral entanglement score and centrality.
    Returns: (centrality_dict, global_score)
    Note: centrality_dict keys will be indices 0..N-1 if adj_matrix
          has no columns.
    """
    try:
        eigvals, eigvecs = np.linalg.eigh(adj_matrix)
        top_eig = np.max(eigvals)
        n = adj_matrix.shape[0]
        if n == 0:
            return {}, 0.0

        global_index = top_eig / n

        top_vec = eigvecs[:, -1]
        centrality = (top_vec ** 2) / np.sum(top_vec ** 2)

        # We return a dict with integer keys here as we don't have column names
        # The caller must map them back if needed.
        return (
            {i: float(c) for i, c in enumerate(centrality)},
            float(global_index)
        )

    except np.linalg.LinAlgError:
        return {}, 0.0


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

        adj = build_entanglement_matrix(returns, self.q)
        centrality_map, global_index = entanglement_indices(adj)

        # Map integer keys back to column names
        asset_scores = {
            returns.columns[i]: score
            for i, score in centrality_map.items()
        }

        breach = global_index > self.threshold

        return EntanglementReport(
            global_index=float(global_index),
            asset_centrality=asset_scores,
            adjacency_matrix_id=f"adj_{len(returns)}",
            threshold_breach=breach
        )
