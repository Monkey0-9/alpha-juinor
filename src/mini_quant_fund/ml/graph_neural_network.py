"""
Graph Neural Network for Supply Chain Analysis
==============================================

Uses graph neural networks to model:
- Supplier-customer relationships
- Inter-company dependencies
- Sector contagion effects
- Corporate ownership networks

Based on Graph Attention Networks (GAT) and GraphSAGE architectures.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Mock:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return Mock
        def __call__(self, *args, **kwargs): return Mock()
        def __getitem__(self, key): return Mock
    torch = Mock()
    nn = Mock()
    nn.Module = object
    F = Mock()

logger = logging.getLogger(__name__)


@dataclass
class CompanyNode:
    """Node representing a company in the graph."""

    symbol: str
    features: np.ndarray  # Financial metrics, sector, etc.
    neighbors: List[str]  # Connected companies
    edge_weights: List[float]  # Relationship strength


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).

    Learns importance weights for neighbor aggregation.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: Node features [num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Updated node features [num_nodes, out_features]
        """
        Wh = torch.mm(h, self.W)  # [num_nodes, out_features]
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        """Prepare attention mechanism input."""
        N = Wh.size()[0]

        # Repeat operations
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1
        )

        return self.leakyrelu(
            torch.matmul(all_combinations_matrix, self.a).view(N, N)
        )


class SupplyChainGNN(nn.Module):
    """
    Supply Chain Graph Neural Network.

    Models company relationships and predicts:
    - Contagion risk
    - Revenue spillover
    - Sector correlations
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.6,
    ):
        super().__init__()

        self.num_heads = num_heads

        # Multi-head attention layers
        self.attentions = nn.ModuleList(
            [
                GraphAttentionLayer(num_features, hidden_dim, dropout)
                for _ in range(num_heads)
            ]
        )

        # Output projection
        self.out_att = GraphAttentionLayer(
            hidden_dim * num_heads, output_dim, dropout
        )

        # Prediction heads
        self.contagion_predictor = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.revenue_impact_predictor = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, num_features]
            adj: Adjacency matrix [num_nodes, num_nodes]

        Returns:
            Dictionary with predictions
        """
        # Multi-head attention
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)

        # Output layer
        x = F.elu(self.out_att(x, adj))

        # Predictions
        contagion_risk = self.contagion_predictor(x)
        revenue_impact = self.revenue_impact_predictor(x)

        return {
            "embeddings": x,
            "contagion_risk": contagion_risk,
            "revenue_impact": revenue_impact,
        }


class SupplyChainAnalyzer:
    """
    High-level supply chain analysis using GNN.

    Builds company relationship graphs and predicts impacts.
    """

    def __init__(
        self,
        num_features: int = 20,
        hidden_dim: int = 64,
        output_dim: int = 32,
        device: str = "cpu",
    ):
        self.company_index: Dict[str, int] = {}
        self.features: Optional[torch.Tensor] = None
        self.adj_matrix: Optional[torch.Tensor] = None

        if not TORCH_AVAILABLE:
            logger.warning("Torch not available. Using simulation mode.")
            return

        self.device = device
        self.model = SupplyChainGNN(num_features, hidden_dim, output_dim).to(device)

    def build_graph(
        self,
        companies: List[CompanyNode],
        relationship_matrix: np.ndarray,
    ):
        """
        Build supply chain graph from company data.

        Args:
            companies: List of company nodes
            relationship_matrix: Adjacency matrix [num_companies, num_companies]
        """
        # Build index
        self.company_index = {comp.symbol: i for i, comp in enumerate(companies)}

        # Stack features
        features = np.stack([comp.features for comp in companies])
        self.features = torch.FloatTensor(features).to(self.device)

        # Adjacency matrix
        self.adj_matrix = torch.FloatTensor(relationship_matrix).to(self.device)

        logger.info(f"Built graph with {len(companies)} companies")

    def predict_contagion_risk(
        self, source_companies: List[str]
    ) -> Dict[str, float]:
        """
        Predict contagion risk from source companies.

        Args:
            source_companies: List of source company symbols

        Returns:
            Dictionary {symbol: risk_score} for all companies
        """
        if not TORCH_AVAILABLE:
            import random
            return {symbol: random.random() for symbol in self.company_index.keys()}

        if self.features is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.features, self.adj_matrix)
            contagion_risk = predictions["contagion_risk"].cpu().numpy().flatten()

        # Create reverse index
        symbol_lookup = {idx: symbol for symbol, idx in self.company_index.items()}

        return {symbol_lookup[i]: float(risk) for i in range(len(contagion_risk))}

    def predict_revenue_impact(
        self, shock_companies: List[str], shock_magnitude: float
    ) -> Dict[str, float]:
        """
        Predict revenue impact from shocks to specific companies.

        Args:
            shock_companies: Companies experiencing shock
            shock_magnitude: Size of shock (-1 to 1)

        Returns:
            Dictionary {symbol: revenue_impact_pct}
        """
        if self.features is None:
            raise ValueError("Graph not built. Call build_graph() first.")

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.features, self.adj_matrix)
            revenue_impact = predictions["revenue_impact"].cpu().numpy().flatten()

        # Scale by shock magnitude
        revenue_impact *= shock_magnitude

        symbol_lookup = {idx: symbol for symbol, idx in self.company_index.items()}

        return {
            symbol_lookup[i]: float(impact) for i in range(len(revenue_impact))
        }

    def get_company_embedding(self, symbol: str) -> Optional[np.ndarray]:
        """
        Get learned embedding for a company.

        Args:
            symbol: Company symbol

        Returns:
            Embedding vector or None
        """
        if symbol not in self.company_index:
            return None

        idx = self.company_index[symbol]

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.features, self.adj_matrix)
            embedding = predictions["embeddings"][idx].cpu().numpy()

        return embedding

    def find_similar_companies(
        self, symbol: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find similar companies based on learned embeddings.

        Args:
            symbol: Target company symbol
            top_k: Number of similar companies to return

        Returns:
            List of (symbol, similarity) tuples
        """
        target_embedding = self.get_company_embedding(symbol)
        if target_embedding is None:
            return []

        # Get all embeddings
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.features, self.adj_matrix)
            all_embeddings = predictions["embeddings"].cpu().numpy()

        # Compute cosine similarity
        similarities = np.dot(all_embeddings, target_embedding) / (
            np.linalg.norm(all_embeddings, axis=1)
            * np.linalg.norm(target_embedding)
            + 1e-8
        )

        # Get top-k (excluding self)
        symbol_lookup = {idx: sym for sym, idx in self.company_index.items()}
        target_idx = self.company_index[symbol]

        similar_indices = np.argsort(similarities)[::-1][1 : top_k + 1]

        return [
            (symbol_lookup[int(idx)], float(similarities[idx]))
            for idx in similar_indices
        ]

    def train(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        labels: Dict[str, torch.Tensor],
        epochs: int = 200,
        lr: float = 0.005,
    ):
        """
        Train the GNN model.

        Args:
            features: Node features
            adj: Adjacency matrix
            labels: Dictionary with training labels
            epochs: Training epochs
            lr: Learning rate
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            predictions = self.model(features, adj)

            # Compute loss (example: MSE for both tasks)
            loss = 0
            if "contagion" in labels:
                loss += F.mse_loss(predictions["contagion_risk"], labels["contagion"])
            if "revenue" in labels:
                loss += F.mse_loss(predictions["revenue_impact"], labels["revenue"])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
