"""
ml/feature_registry.py

Feature Registry & Schema Enforcement (Ticket 6)

Central registry for all ML features with:
- Schema versioning
- Hash-based validation
- Drift detection hooks
- Training/inference consistency enforcement

Features must be registered here before use in production models.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger("FEATURE_REGISTRY")


class FeatureStatus(str, Enum):
    """Feature lifecycle status."""
    DRAFT = "DRAFT"           # In development
    EXPERIMENTAL = "EXPERIMENTAL"  # Testing
    PRODUCTION = "PRODUCTION"     # Approved for prod
    DEPRECATED = "DEPRECATED"     # Being phased out
    RETIRED = "RETIRED"           # No longer used


@dataclass
class FeatureSpec:
    """Specification for a registered feature."""
    name: str
    version: str
    dtype: str                    # "float64", "int64", "bool", etc.
    description: str
    schema_hash: str              # MD5 of feature spec
    status: FeatureStatus
    validation_min: Optional[float] = None
    validation_max: Optional[float] = None
    allow_null: bool = False
    fill_value: Optional[float] = None
    compute_fn: Optional[str] = None  # Python path to compute function
    dependencies: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.utcnow().isoformat() + 'Z'
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSpec":
        status = d.get('status', 'DRAFT')
        if isinstance(status, str):
            try:
                status = FeatureStatus(status)
            except ValueError:
                status = FeatureStatus.DRAFT

        return cls(
            name=d['name'],
            version=d.get('version', 'v1.0.0'),
            dtype=d.get('dtype', 'float64'),
            description=d.get('description', ''),
            schema_hash=d.get('schema_hash', ''),
            status=status,
            validation_min=d.get('validation_min'),
            validation_max=d.get('validation_max'),
            allow_null=d.get('allow_null', False),
            fill_value=d.get('fill_value'),
            compute_fn=d.get('compute_fn'),
            dependencies=d.get('dependencies', []),
            created_at=d.get('created_at', ''),
            updated_at=d.get('updated_at', '')
        )


@dataclass
class FeatureSetSchema:
    """Schema for a set of features used by a model."""
    model_id: str
    version: str
    features: List[str]           # Ordered feature names
    schema_hash: str              # Hash of feature specs
    frozen: bool = False          # If True, no changes allowed
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + 'Z'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureRegistry:
    """
    Central Feature Registry.

    All ML features MUST be registered here before use.
    Provides:
    - Feature versioning
    - Schema validation
    - Training/inference consistency
    - Drift detection hooks

    Usage:
        registry = FeatureRegistry()
        registry.register_feature(FeatureSpec(...))
        schema_hash = registry.freeze_schema_for_model(model_id, features)
        is_valid = registry.validate_input(model_id, df)
    """

    # Standard features (pre-registered)
    STANDARD_FEATURES = [
        ("returns_1d", "float64", "1-day returns"),
        ("returns_5d", "float64", "5-day returns"),
        ("returns_20d", "float64", "20-day returns"),
        ("volatility_10d", "float64", "10-day rolling volatility"),
        ("volatility_20d", "float64", "20-day rolling volatility"),
        ("volume_ratio", "float64", "Volume / 20-day avg volume"),
        ("rsi_14", "float64", "14-day RSI"),
        ("macd", "float64", "MACD line"),
        ("macd_signal", "float64", "MACD signal line"),
        ("bb_position", "float64", "Position within Bollinger Bands [0,1]"),
        ("atr_14", "float64", "14-day Average True Range"),
        ("vwap_deviation", "float64", "Deviation from VWAP"),
        ("momentum_10d", "float64", "10-day price momentum"),
        ("momentum_20d", "float64", "20-day price momentum"),
    ]

    def __init__(self, db_manager=None):
        """
        Initialize FeatureRegistry.

        Args:
            db_manager: DatabaseManager instance
        """
        self._db = db_manager
        self._features: Dict[str, FeatureSpec] = {}
        self._model_schemas: Dict[str, FeatureSetSchema] = {}

        # Register standard features
        self._register_standard_features()

    @property
    def db(self):
        if self._db is None:
            from database.manager import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    def _register_standard_features(self):
        """Register standard features."""
        for name, dtype, desc in self.STANDARD_FEATURES:
            spec = FeatureSpec(
                name=name,
                version="v1.0.0",
                dtype=dtype,
                description=desc,
                schema_hash=self._compute_feature_hash(name, dtype),
                status=FeatureStatus.PRODUCTION,
                allow_null=False,
                fill_value=0.0
            )
            self._features[name] = spec

    def _compute_feature_hash(self, name: str, dtype: str, version: str = "v1.0.0") -> str:
        """Compute hash for a feature spec."""
        content = json.dumps({
            "name": name,
            "dtype": dtype,
            "version": version
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def register_feature(self, spec: FeatureSpec) -> bool:
        """
        Register a new feature.

        Args:
            spec: FeatureSpec to register

        Returns:
            True if registered, False if already exists with different version
        """
        # Compute hash if not provided
        if not spec.schema_hash:
            spec.schema_hash = self._compute_feature_hash(
                spec.name, spec.dtype, spec.version
            )

        # Check for conflicts
        if spec.name in self._features:
            existing = self._features[spec.name]
            if existing.schema_hash != spec.schema_hash and existing.status == FeatureStatus.PRODUCTION:
                logger.error(f"Feature {spec.name} already exists with different schema")
                return False

        self._features[spec.name] = spec

        # Persist to database
        self._persist_feature(spec)

        logger.info(json.dumps({
            "event": "FEATURE_REGISTERED",
            "name": spec.name,
            "version": spec.version,
            "status": spec.status.value,
            "hash": spec.schema_hash
        }))

        return True

    def _persist_feature(self, spec: FeatureSpec):
        """Persist feature to database."""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO feature_registry
                    (feature_name, version, schema_hash, dtype, validation_fn, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    spec.name,
                    spec.version,
                    spec.schema_hash,
                    spec.dtype,
                    spec.compute_fn,
                    spec.description,
                    spec.created_at,
                    spec.updated_at
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist feature {spec.name}: {e}")

    def get_feature(self, name: str) -> Optional[FeatureSpec]:
        """Get a registered feature."""
        return self._features.get(name)

    def list_features(self, status: FeatureStatus = None) -> List[FeatureSpec]:
        """List registered features, optionally filtered by status."""
        if status:
            return [f for f in self._features.values() if f.status == status]
        return list(self._features.values())

    def freeze_schema_for_model(
        self,
        model_id: str,
        features: List[str],
        version: str = "v1.0.0"
    ) -> str:
        """
        Freeze feature schema for a model.

        Once frozen, the model's feature list cannot change.
        Returns the schema hash.
        """
        # Verify all features are registered
        missing = [f for f in features if f not in self._features]
        if missing:
            raise ValueError(f"Unregistered features: {missing}")

        # Compute schema hash from feature specs
        feature_hashes = [self._features[f].schema_hash for f in features]
        schema_content = json.dumps({
            "model_id": model_id,
            "features": features,
            "feature_hashes": feature_hashes,
            "version": version
        }, sort_keys=True)
        schema_hash = hashlib.md5(schema_content.encode()).hexdigest()

        # Create frozen schema
        schema = FeatureSetSchema(
            model_id=model_id,
            version=version,
            features=features,
            schema_hash=schema_hash,
            frozen=True
        )

        self._model_schemas[model_id] = schema

        logger.info(json.dumps({
            "event": "SCHEMA_FROZEN",
            "model_id": model_id,
            "n_features": len(features),
            "schema_hash": schema_hash
        }))

        return schema_hash

    def get_model_schema(self, model_id: str) -> Optional[FeatureSetSchema]:
        """Get frozen schema for a model."""
        return self._model_schemas.get(model_id)

    def validate_input(
        self,
        model_id: str,
        df: pd.DataFrame,
        raise_on_error: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate input DataFrame against model's frozen schema.

        Args:
            model_id: Model identifier
            df: Input DataFrame to validate
            raise_on_error: If True, raise exception on validation failure

        Returns:
            Tuple of (is_valid, list of errors)
        """
        schema = self._model_schemas.get(model_id)
        if not schema:
            error = f"No frozen schema for model {model_id}"
            if raise_on_error:
                raise ValueError(error)
            return False, [error]

        errors = []

        # Check for missing features
        missing = [f for f in schema.features if f not in df.columns]
        if missing:
            errors.append(f"Missing features: {missing}")

        # Check for extra features (warning only)
        extra = [f for f in df.columns if f not in schema.features]
        if extra:
            logger.warning(f"Extra features in input (will be ignored): {extra}")

        # Validate dtypes and ranges for present features
        for feat_name in schema.features:
            if feat_name not in df.columns:
                continue

            spec = self._features.get(feat_name)
            if not spec:
                continue

            col = df[feat_name]

            # Check nulls
            if not spec.allow_null and col.isnull().any():
                null_count = col.isnull().sum()
                errors.append(f"{feat_name}: has {null_count} null values")

            # Check range
            if spec.validation_min is not None:
                below_min = (col < spec.validation_min).sum()
                if below_min > 0:
                    errors.append(f"{feat_name}: {below_min} values below min {spec.validation_min}")

            if spec.validation_max is not None:
                above_max = (col > spec.validation_max).sum()
                if above_max > 0:
                    errors.append(f"{feat_name}: {above_max} values above max {spec.validation_max}")

            # Check for inf
            if col.dtype in ['float64', 'float32']:
                inf_count = np.isinf(col).sum()
                if inf_count > 0:
                    errors.append(f"{feat_name}: has {inf_count} infinite values")

        is_valid = len(errors) == 0

        if not is_valid:
            logger.error(json.dumps({
                "event": "SCHEMA_VALIDATION_FAILED",
                "model_id": model_id,
                "errors": errors[:10]  # Limit to first 10 errors
            }))

            if raise_on_error:
                raise ValueError(f"Schema validation failed: {errors}")

        return is_valid, errors

    def compute_input_hash(self, df: pd.DataFrame, features: List[str]) -> str:
        """
        Compute hash of input data for reproducibility.

        Used to verify that training and inference data have same schema.
        """
        # Hash column names and dtypes
        col_info = {f: str(df[f].dtype) if f in df.columns else "MISSING" for f in features}
        content = json.dumps(col_info, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def prepare_input(
        self,
        model_id: str,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare input DataFrame for model inference.

        - Selects only registered features in correct order
        - Fills nulls with registered fill values
        - Validates schema

        Args:
            model_id: Model identifier
            df: Raw input DataFrame

        Returns:
            Prepared DataFrame with only required features in correct order
        """
        schema = self._model_schemas.get(model_id)
        if not schema:
            raise ValueError(f"No frozen schema for model {model_id}")

        # Select and order features
        result = pd.DataFrame()
        for feat_name in schema.features:
            if feat_name in df.columns:
                result[feat_name] = df[feat_name].copy()
            else:
                # Use fill value if available
                spec = self._features.get(feat_name)
                fill_val = spec.fill_value if spec else 0.0
                result[feat_name] = fill_val
                logger.warning(f"Missing feature {feat_name}, filled with {fill_val}")

        # Fill nulls
        for feat_name in schema.features:
            spec = self._features.get(feat_name)
            if spec and spec.fill_value is not None:
                result[feat_name] = result[feat_name].fillna(spec.fill_value)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_status = {}
        for f in self._features.values():
            status = f.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_features": len(self._features),
            "by_status": by_status,
            "frozen_models": len(self._model_schemas),
            "standard_features": len(self.STANDARD_FEATURES)
        }


# Singleton instance
_instance: Optional[FeatureRegistry] = None


def get_feature_registry() -> FeatureRegistry:
    """Get singleton FeatureRegistry instance."""
    global _instance
    if _instance is None:
        _instance = FeatureRegistry()
    return _instance
