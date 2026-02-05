import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import json
import os
import sys

from database.manager import DatabaseManager
from database.schema import FeatureRecord

logger = logging.getLogger("FEATURE_STORE")

class FeatureStore:
    """
    Institutional Feature & Representation Store.
    Enforces feature schema freezing and provides versioned access.
    """
    def __init__(self, schema_path: str = "configs/feature_schema.json"):
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.db = DatabaseManager()
        self.version = self.schema.get("version", "1.0.0")

    def _load_schema(self) -> Dict[str, Any]:
        try:
            with open(self.schema_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load feature schema from {self.schema_path}: {e}")
            return {"version": "0.0.0", "features": []}

    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Frozen schema validation.
        Ensures production models only see features they were trained on.
        """
        required_features = [f["name"] for f in self.schema.get("features", [])]
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            logger.error(f"FEATURE_SCHEMA_VIOLATION: Missing {len(missing)} features: {missing}")
            return False
        return True

    def compute_and_store(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Compute features for a symbol and store in DB if valid.
        """
        try:
            # 1. Compute features (modern functional API)
            from data.processors.features import compute_features_for_symbol
            features_df = compute_features_for_symbol(data)
            if features_df is None or features_df.empty:
                return False

            # 2. Validate against schema
            if not self.validate_features(features_df):
                return False

            # 3. Persist latest feature vector
            latest_date = features_df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            latest_features = features_df.iloc[-1].to_dict()

            # Filter to schema only if enforced
            required_names = [f["name"] for f in self.schema.get("features", [])]
            filtered_features = {k: v for k, v in latest_features.items() if k in required_names}

            record = FeatureRecord(
                symbol=symbol,
                date=latest_date,
                features=filtered_features,
                version=self.version
            )

            self.db.upsert_features([record])
            logger.info(f"FEATURE_STORE: Successfully updated features for {symbol} (date: {latest_date})")
            return True

        except Exception as e:
            logger.exception(f"Feature computation/storage failed for {symbol}: {e}")
            return False

    def get_latest(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Retrieve latest features from DB."""
        return self.db.get_latest_features(symbols)
