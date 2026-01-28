"""
utils/errors.py
Custom exception classes for institutional governance and ML operations
"""
from typing import List, Optional


class FeatureValidationError(Exception):
    """Raised when feature validation fails."""
    pass


class ModelFeatureMismatchError(Exception):
    """
    Raised when model expects different features than provided at runtime.

    This is a CRITICAL governance error that should trigger ML_ALPHA disable.
    """

    def __init__(
        self,
        message: str,
        expected_features: Optional[List[str]] = None,
        provided_features: Optional[List[str]] = None,
        missing: Optional[List[str]] = None,
        extra: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.expected_features = expected_features or []
        self.provided_features = provided_features or []
        self.missing = missing or []
        self.extra = extra or []

    def to_dict(self):
        """Convert to structured dict for JSON logging."""
        return {
            "error_type": "MODEL_FEATURE_MISMATCH",
            "message": str(self),
            "expected_count": len(self.expected_features),
            "provided_count": len(self.provided_features),
            "missing_features": self.missing,
            "extra_features": self.extra
        }


class GovernanceDisabledError(Exception):
    """Raised when attempting ML operations while governance has disabled the model."""
    pass
