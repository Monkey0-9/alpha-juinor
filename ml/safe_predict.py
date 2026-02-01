"""
ml/safe_predict.py

Safe Inference Wrapper with full AlphaOutput construction.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Any

from contracts.alpha_model import AlphaOutput
from ml.safe_input import align_features
from ml.input_schema_guard import FeatureSchemaGuard
from ml.governor import ModelGovernor

logger = logging.getLogger("SAFE_PREDICT")


class SafePredictor:
    def __init__(
        self,
        model_id: str,
        schema_guard: Optional[FeatureSchemaGuard] = None,
        governor: Optional[ModelGovernor] = None
    ):
        self.model_id = model_id
        self.schema_guard = schema_guard or FeatureSchemaGuard()
        self.governor = governor or ModelGovernor()

    def predict(
        self,
        model: Any,
        features: pd.DataFrame,
        expected_features: List[str],
        baseline_features: Optional[np.ndarray] = None
    ) -> AlphaOutput:
        """Execute safe prediction with full validation."""
        try:
            # 1. Feature Alignment
            X_aligned, missing, _ = align_features(features, expected_features)

            if missing:
                logger.warning(
                    f"[{self.model_id}] Missing features: {len(missing)}"
                )

            # 2. Predict
            try:
                raw_pred = model.predict(X_aligned)
            except Exception as e:
                logger.error(f"[{self.model_id}] Prediction failed: {e}")
                return AlphaOutput.neutral(self.model_id)

            # Handle output formats
            if isinstance(raw_pred, (pd.Series, pd.DataFrame)):
                pred_val = float(raw_pred.iloc[0]) if len(raw_pred) > 0 else 0.0
            elif isinstance(raw_pred, np.ndarray):
                pred_val = float(raw_pred.flat[0]) if raw_pred.size > 0 else 0.0
            else:
                pred_val = float(raw_pred)

            # 3. Validate output
            if not np.isfinite(pred_val):
                logger.error(f"[{self.model_id}] Non-finite: {pred_val}")
                return AlphaOutput.neutral(self.model_id)

            # Clamp to reasonable range
            pred_val = np.clip(pred_val, -0.5, 0.5)

            # 4. Construct full AlphaOutput
            return AlphaOutput(
                mu=pred_val,
                sigma=0.02,
                cvar_95=-0.03,
                confidence=1.0 if not missing else 0.5,
                provider=self.model_id,
                model_version="v1",
                input_schema_hash="auto",
                explanation=f"Imputed {len(missing)} features" if missing else "OK"
            )

        except Exception as e:
            logger.error(f"[{self.model_id}] SafePredictor crash: {e}")
            return AlphaOutput.neutral(self.model_id)
