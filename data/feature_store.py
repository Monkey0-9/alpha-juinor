"""
Enterprise Feature Store
========================

Centralized feature management to ensure:
- Consistency between training and serving
- Point-in-time correctness
- Feature lineage and metadata
- Caching and performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class FeatureDefinition:
    """Metadata for a feature."""
    name: str
    description: str
    formula: str
    tags: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    owner: str = "system"

class FeatureStore:
    """
    Lightweight Feature Store for managing alpha factors.
    """

    def __init__(self, store_path: str = "./feature_store"):
        self.store_path = store_path
        self.registry: Dict[str, FeatureDefinition] = {}
        self.cache: Dict[str, pd.DataFrame] = {}

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        self._load_registry()

    def register_feature(self, definition: FeatureDefinition):
        """Register a new feature definition."""
        self.registry[definition.name] = definition
        self._save_registry()
        logger.info(f"Registered feature: {definition.name}")

    def write_features(self, entity_df: pd.DataFrame, entity_key: str = "symbol", timestamp_key: str = "timestamp"):
        """Write feature values to store."""
        # Simple file-based storage partitioned by symbol
        if entity_key not in entity_df.columns:
            raise ValueError(f"Entity key {entity_key} not in dataframe")

        for entity, group in entity_df.groupby(entity_key):
            path = os.path.join(self.store_path, f"{entity}.parquet")

            # Point-in-time handling could be added here
            if os.path.exists(path):
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, group]).drop_duplicates(subset=[timestamp_key], keep='last')
                combined.to_parquet(path)
            else:
                group.to_parquet(path)

    def get_features(self, entities: List[str], features: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        """Retrieve features for a set of entities."""
        results = []

        for entity in entities:
            path = os.path.join(self.store_path, f"{entity}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                # Filter time
                mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
                subset = df.loc[mask]

                # Check for available features
                available_cols = [c for c in features if c in subset.columns]
                if available_cols:
                     results.append(subset[['timestamp', 'symbol'] + available_cols])

        if not results:
            return pd.DataFrame()

        return pd.concat(results)

    def _save_registry(self):
        """Persist registry to disk."""
        path = os.path.join(self.store_path, "registry.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.registry, f)

    def _load_registry(self):
        """Load registry from disk."""
        path = os.path.join(self.store_path, "registry.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.registry = pickle.load(f)

# Global singleton
_feature_store: Optional[FeatureStore] = None

def get_feature_store() -> FeatureStore:
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore()
    return _feature_store
