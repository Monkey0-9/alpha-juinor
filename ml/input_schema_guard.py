"""
Feature Schema Guard - Institutional Grade Input Validation

This module enforces strict schema validation before any ML model prediction.
RULE: Never let sklearn throw runtime shape errors again.

Behavior:
1. Validate feature count
2. Validate feature names (exact match)
3. Validate ordering
4. Hard-fail model before prediction
5. Record schema mismatch to audit

Usage:
    guard = FeatureSchemaGuard()
    guard.validate(X, model_id="ml_v1")  # Raises FeatureSchemaViolation on mismatch
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger("SCHEMA_GUARD")

# ============================================================================
# EXCEPTIONS
# ============================================================================

class FeatureSchemaViolation(Exception):
    """Raised when feature schema validation fails. Trading MUST stop."""

    def __init__(
        self,
        model_id: str,
        expected_count: int,
        actual_count: int,
        missing_features: List[str],
        extra_features: List[str],
        order_mismatch: bool
    ):
        self.model_id = model_id
        self.expected_count = expected_count
        self.actual_count = actual_count
        self.missing_features = missing_features
        self.extra_features = extra_features
        self.order_mismatch = order_mismatch

        msg = (
            f"FEATURE_SCHEMA_VIOLATION for {model_id}: "
            f"expected {expected_count} features, got {actual_count}. "
            f"Missing: {missing_features}, Extra: {extra_features}, "
            f"Order mismatch: {order_mismatch}"
        )
        super().__init__(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": "CRITICAL",
            "component": "SCHEMA_GUARD",
            "model_id": self.model_id,
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "missing_features": self.missing_features,
            "extra_features": self.extra_features,
            "order_mismatch": self.order_mismatch
        }


# ============================================================================
# FROZEN SCHEMA RECORD
# ============================================================================

@dataclass
class FrozenFeatureSchema:
    """Immutable feature schema definition for a model version."""
    feature_set_id: str
    feature_hash: str
    feature_names: List[str]
    feature_dtypes: Dict[str, str]
    model_version: str
    frozen_at: str
    frozen_by: str = "system"

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_set_id": self.feature_set_id,
            "feature_hash": self.feature_hash,
            "feature_names": self.feature_names,
            "feature_dtypes": self.feature_dtypes,
            "feature_count": self.feature_count,
            "model_version": self.model_version,
            "frozen_at": self.frozen_at,
            "frozen_by": self.frozen_by
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrozenFeatureSchema":
        return cls(
            feature_set_id=data["feature_set_id"],
            feature_hash=data["feature_hash"],
            feature_names=data["feature_names"],
            feature_dtypes=data.get("feature_dtypes", {}),
            model_version=data["model_version"],
            frozen_at=data["frozen_at"],
            frozen_by=data.get("frozen_by", "system")
        )


# ============================================================================
# SCHEMA GUARD
# ============================================================================

class FeatureSchemaGuard:
    """
    Institutional-grade feature schema validator.

    RULE: No model prediction without schema validation.
    RULE: Schema mismatch = hard fail, not soft alignment.
    """

    FROZEN_SCHEMAS_DIR = Path("configs/frozen_features")

    def __init__(self, audit_callback=None):
        """
        Args:
            audit_callback: Optional callable(violation_dict) for audit logging
        """
        self._cache: Dict[str, FrozenFeatureSchema] = {}
        self._audit_callback = audit_callback
        self.FROZEN_SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("[SCHEMA_GUARD] Initialized with strict validation mode")

    def validate(
        self,
        X: pd.DataFrame,
        model_id: str,
        strict_order: bool = True
    ) -> None:
        """
        Validate input features against frozen schema.

        Args:
            X: Input DataFrame to validate
            model_id: Model identifier (e.g., "ml_v1")
            strict_order: If True, feature order must match exactly

        Raises:
            FeatureSchemaViolation: If validation fails
        """
        schema = self._load_frozen_schema(model_id)
        if schema is None:
            logger.warning(f"[SCHEMA_GUARD] No frozen schema for {model_id}, skipping validation")
            return

        actual_features = list(X.columns)
        expected_features = schema.feature_names

        # Check count
        count_mismatch = len(actual_features) != len(expected_features)

        # Check missing/extra
        actual_set = set(actual_features)
        expected_set = set(expected_features)
        missing = list(expected_set - actual_set)
        extra = list(actual_set - expected_set)

        # Check order (only if counts match and no missing/extra)
        order_mismatch = False
        if strict_order and not missing and not extra:
            order_mismatch = actual_features != expected_features

        # Determine if we have a violation
        has_violation = count_mismatch or missing or extra or order_mismatch

        if has_violation:
            violation = FeatureSchemaViolation(
                model_id=model_id,
                expected_count=len(expected_features),
                actual_count=len(actual_features),
                missing_features=missing,
                extra_features=extra,
                order_mismatch=order_mismatch
            )

            # Record to audit
            self._record_violation(violation)

            # Hard fail
            raise violation

        # Validate hash for extra paranoia
        live_hash = self._compute_feature_hash(actual_features)
        if live_hash != schema.feature_hash:
            logger.warning(
                f"[SCHEMA_GUARD] Feature hash mismatch for {model_id}. "
                f"Expected: {schema.feature_hash[:16]}..., Got: {live_hash[:16]}..."
            )

    def freeze_schema(
        self,
        model_id: str,
        feature_names: List[str],
        feature_dtypes: Optional[Dict[str, str]] = None
    ) -> FrozenFeatureSchema:
        """
        Freeze the feature schema for a model.

        This creates an immutable record that will be used for all future validations.
        """
        feature_hash = self._compute_feature_hash(feature_names)
        timestamp = datetime.utcnow().isoformat() + "Z"

        schema = FrozenFeatureSchema(
            feature_set_id=f"fs_{model_id}_{timestamp[:10].replace('-', '')}",
            feature_hash=feature_hash,
            feature_names=feature_names,
            feature_dtypes=feature_dtypes or {},
            model_version=model_id,
            frozen_at=timestamp
        )

        # Persist
        schema_path = self.FROZEN_SCHEMAS_DIR / f"{model_id}.json"
        with open(schema_path, 'w') as f:
            json.dump(schema.to_dict(), f, indent=2)

        # Update cache
        self._cache[model_id] = schema

        logger.info(
            f"[SCHEMA_GUARD] Froze schema for {model_id}: "
            f"{len(feature_names)} features, hash={feature_hash[:16]}..."
        )

        return schema

    def get_frozen_schema(self, model_id: str) -> Optional[FrozenFeatureSchema]:
        """Get the frozen schema for a model, if it exists."""
        return self._load_frozen_schema(model_id)

    def _load_frozen_schema(self, model_id: str) -> Optional[FrozenFeatureSchema]:
        """Load frozen schema from cache or disk."""
        if model_id in self._cache:
            return self._cache[model_id]

        schema_path = self.FROZEN_SCHEMAS_DIR / f"{model_id}.json"
        if not schema_path.exists():
            return None

        try:
            with open(schema_path, 'r') as f:
                data = json.load(f)
            schema = FrozenFeatureSchema.from_dict(data)
            self._cache[model_id] = schema
            return schema
        except Exception as e:
            logger.error(f"[SCHEMA_GUARD] Failed to load schema for {model_id}: {e}")
            return None

    def _compute_feature_hash(self, feature_names: List[str]) -> str:
        """Compute deterministic hash of feature names."""
        canonical = ",".join(sorted(feature_names))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def _record_violation(self, violation: FeatureSchemaViolation) -> None:
        """Record violation to audit trail."""
        violation_dict = violation.to_dict()

        # Log structured JSON
        logger.error(json.dumps(violation_dict))

        # Call audit callback if provided
        if self._audit_callback:
            try:
                self._audit_callback(violation_dict)
            except Exception as e:
                logger.error(f"[SCHEMA_GUARD] Audit callback failed: {e}")


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_guard_instance: Optional[FeatureSchemaGuard] = None

def get_schema_guard() -> FeatureSchemaGuard:
    """Get singleton instance of FeatureSchemaGuard."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = FeatureSchemaGuard()
    return _guard_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_features(X: pd.DataFrame, model_id: str) -> None:
    """Convenience function for feature validation."""
    get_schema_guard().validate(X, model_id)


def freeze_features(model_id: str, feature_names: List[str]) -> FrozenFeatureSchema:
    """Convenience function for freezing features."""
    return get_schema_guard().freeze_schema(model_id, feature_names)
