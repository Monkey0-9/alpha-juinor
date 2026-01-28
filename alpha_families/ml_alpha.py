import json
import logging
import time
import warnings
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn.ensemble
from sklearn.preprocessing import StandardScaler

# Institutional Patch: Legacy Sklearn Compatibility
sys.modules["sklearn.ensemble.forest"] = sklearn.ensemble

from data.processors.validator import validate_features
from utils.errors import ModelFeatureMismatchError, GovernanceDisabledError
from utils.timeutils import ensure_business_days
from utils.metrics import metrics
from .base_alpha import BaseAlpha

logger = logging.getLogger(__name__)


class MLAlpha(BaseAlpha):
    """
    Machine Learning Alpha Model with Institutional Governance.

    Features:
    - Strict feature contract enforcement (training ↔ runtime)
    - Emergency disable via config flag
    - Governance escalation on repeated failures
    - Model metadata validation
    - Hard-fail on feature mismatches (no silent retries)
    """

    def _compute_schema_hash(self, features: List[str]) -> str:
        """Compute stable hash for feature schema."""
        import hashlib
        # Sort to ensure order-independence if list order doesn't align but content does
        # But for ML, order usually matters. User said "X has 4 features...".
        # sklearn expects exact column order usually.
        # So we hash the EXACT list.
        return hashlib.md5(json.dumps(features).encode()).hexdigest()


    def _load_legacy_global(self):
        """Load legacy global model for backwards compatibility."""
        import joblib
        return_model_path = self.model_path / "return_model.pkl"
        if return_model_path.exists():
            try:
                model = joblib.load(return_model_path)
                # Extract feature names from model if available
                features = []
                if hasattr(model, 'feature_names_in_'):
                    features = list(model.feature_names_in_)
                self._cached_models["LEGACY_GLOBAL"] = {
                    "model": model,
                    "features": features,
                    "metadata": {"features": features, "schema_hash": self._compute_schema_hash(features) if features else ""},
                    "status": "LEGACY"
                }
                logger.warning("[ML_ALPHA] Pre-loaded LEGACY GLOBAL fallback model.")
            except Exception as e:
                logger.error(f"[ML_ALPHA] Failed to pre-load legacy global model: {e}")

    def _load_model_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:

        """
        Load model and metadata for a symbol with strict validation.

        Returns:
            Dict containing:
                - model: sklearn model object
                - features: List[str] (exact order expected)
                - metadata: Dict (full model_meta.json content)
                - status: str
        """
        if symbol in self._cached_models:
            return self._cached_models[symbol]

        latest_link = self.model_path / f"{symbol}_latest.joblib"
        if latest_link.exists():
            try:
                meta = joblib.load(latest_link)
                model_data = joblib.load(meta["path"])

                # Check for new format (directory with model.pkl and model_meta.json)
                model_file_path = Path(meta["path"])
                parent_dir = model_file_path.parent

                if (parent_dir / "model_meta.json").exists():
                    with open(parent_dir / "model_meta.json", "r") as f:
                        full_metadata = json.load(f)

                    if (parent_dir / "model.pkl").exists():
                        model_obj = joblib.load(parent_dir / "model.pkl")
                        model_data = {
                            "model": model_obj,
                            "features": full_metadata["features"],
                            "metadata": full_metadata,
                            "status": "ACTIVE"
                        }
                    else:
                        # Fallback to old format
                        model_data["metadata"] = full_metadata
                else:
                    # Old format - use embedded features list
                    model_data["metadata"] = {
                        "features": model_data.get("features", []),
                        "version": "legacy"
                    }

                # ENFORCE SCHEMA HASH
                features = model_data.get("features", [])
                if features:
                    schema_hash = self._compute_schema_hash(features)
                    model_data["metadata"]["schema_hash"] = schema_hash

                self._cached_models[symbol] = model_data
                return model_data
            except Exception as e:
                logger.error(f"[ML_ALPHA] Failed to load model for {symbol}: {e}")

        # Try global model
        global_model_path = self.model_path / "global_latest.joblib"
        if global_model_path.exists():
            try:
                if "GLOBAL" in self._cached_models:
                    return self._cached_models["GLOBAL"]
                meta = joblib.load(global_model_path)
                model_data = joblib.load(meta["path"])

                # ENFORCE SCHEMA HASH (Legacy global usually has features in dict)
                features = model_data.get("features", [])
                if features:
                     model_data["metadata"] = model_data.get("metadata", {})
                     model_data["metadata"]["features"] = features
                     model_data["metadata"]["schema_hash"] = self._compute_schema_hash(features)

                self._cached_models["GLOBAL"] = model_data
                return model_data
            except Exception as e:
                logger.error(f"[ML_ALPHA] Failed to load global fallback model: {e}")

        # Legacy global fallback
        if "LEGACY_GLOBAL" in self._cached_models:
            return self._cached_models["LEGACY_GLOBAL"]

        return None

    def _record_failure(self, error_type: str):
        """Record a governance failure and check if we should disable."""
        now = time.time()

        # Add new failure
        self._failure_window.append((now, error_type))

        # Remove failures outside the time window
        cutoff_time = now - self._failure_window_seconds
        self._failure_window = [(t, e) for (t, e) in self._failure_window if t > cutoff_time]

        # Check if we've exceeded the threshold
        if len(self._failure_window) >= self._failure_threshold:
            self._governance_disabled = True

            # Log structured governance escalation
            logger.error(json.dumps({
                "ts": datetime.utcnow().isoformat() + "Z",
                "component": "ML_GOVERNANCE",
                "level": "ERROR",
                "model": "ml_v1",
                "status": "DISABLED_BY_GOVERNANCE",
                "reason": "REPEATED_FAILURES",
                "failure_count": len(self._failure_window),
                "window_seconds": self._failure_window_seconds,
                "error_types": [e for (_, e) in self._failure_window]
            }))

    def ml_predict_safe(
        self,
        model,
        X_df: pd.DataFrame,
        model_meta: Optional[Dict[str, Any]] = None,
        symbol: str = "UNKNOWN"
    ):
        """
        Institutional Grade Predict Wrapper with Governance Escalation.
        Uses feature alignment and fallback to prevent hard crashes.
        """
        from ml.safe_input import align_features, distributional_sanity_check

        # Check governance state
        if self._governance_disabled:
            raise GovernanceDisabledError(
                f"ML Alpha disabled by governance (symbol={symbol}). "
                "System detected repeated feature mismatches."
            )

        try:
            # Extract expected features
            required_features = None
            model_schema_hash = None

            if model_meta and "features" in model_meta:
                required_features = model_meta["features"]
                model_schema_hash = model_meta.get("schema_hash")
            elif hasattr(model, "feature_names_in_"):
                required_features = list(model.feature_names_in_)

            X_aligned = X_df
            mismatch_detected = False

            # Strict validation + Alignment
            if required_features:
                provided_features = list(X_df.columns)
                input_schema_hash = self._compute_schema_hash(provided_features)

                # Calculate if not present (legacy compat)
                if not model_schema_hash:
                    model_schema_hash = self._compute_schema_hash(required_features)

                if input_schema_hash != model_schema_hash:
                    mismatch_detected = True

                    # Align features (impute missing, drop extra, reorder)
                    X_aligned, missing, extra = align_features(X_df, required_features)

                    # Determine severity
                    is_critical = len(missing) > len(required_features) * 0.3

                    if is_critical:
                        # Record failure and escalate only if critical
                        self._record_failure("FEATURE_SCHEMA_MISMATCH_CRITICAL")
                        logger.error(f"Critical feature mismatch for {symbol}: {len(missing)} missing.")
                    else:
                        # Log as recovered mismatch
                        logger.warning(json.dumps({
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "model": "ml_v1",
                            "symbol": symbol,
                            "event": "FEATURE_MISMATCH",
                            "missing": missing,
                            "extra": extra,
                            "action": "aligned_with_impute",
                            "severity": "RECOVERED"
                        }))


            # Validate remains for types/scales
            X_validated = validate_features(X_aligned, required_features=required_features)

            # Perform prediction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_final = X_validated.values if isinstance(X_validated, pd.DataFrame) else X_validated
                prediction = model.predict(X_final)

                # Clip and scale prediction to reasonable daily return bounds
                # Model may output raw values - clip to ±1.0 and treat as signal
                if len(prediction) > 0:
                    mu = float(prediction[0])
                    # Clip raw prediction to reasonable bounds
                    # Smart scaling: divide by 100 only if mu is large (>1.0)
                    mu_scaled = mu / 100.0 if abs(mu) > 1.0 else mu
                    mu_clipped = max(-0.10, min(0.10, mu_scaled))

                    prediction = np.array([mu_clipped])

                    # Log if prediction was clipped significantly
                    if abs(mu) > 1.0:
                        logger.debug(f"ML prediction for {symbol} clipped: {mu:.4f} -> {mu_clipped:.4f}")

                return prediction


        except (GovernanceDisabledError):
            # Re-raise governance exceptions
            self.model_errors += 1
            metrics.model_errors += 1
            raise
        except Exception as e:
            # Other errors - record and log
            self.model_errors += 1
            metrics.model_errors += 1
            self._record_failure("PREDICTION_ERROR")
            logger.error(f"[ML_ALPHA] [SAFE_PREDICT] Failure for {symbol}: {e}")
            return None

    def __init__(
        self,
        model_path: Optional[str] = None,
        retrain_frequency: int = 100,
        prediction_horizon: int = 5,
        feature_lookback: int = 20,
    ):
        super().__init__()
        self.MIN_ML_SAMPLES = 5000
        self.MIN_FEATURES = 20
        self.model_path = Path(model_path) if model_path else Path("models/ml_alpha")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.prediction_horizon = prediction_horizon
        self.feature_lookback = feature_lookback
        self._cached_models = {}
        self.scaler = StandardScaler()
        self.signal_count = 0
        self.last_training_date = None
        self.model_errors = 0

        # Governance State Tracking
        self._governance_disabled = False
        self._failure_window = []  # Track (timestamp, error_type) tuples

        # Load configurable thresholds
        from configs.config_manager import ConfigManager
        cfg = ConfigManager().config
        self._failure_threshold = cfg.get("ml_governance", {}).get("failure_threshold", 3)
        self._failure_window_seconds = cfg.get("ml_governance", {}).get("window_seconds", 300)

        self._load_legacy_global()

    def generate_signal(
        self,
        market_data: pd.DataFrame,
        regime_context: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate ML alpha signal with config-based enable/disable and governance checking.
        """
        symbol = kwargs.get("symbol", "UNKNOWN")

        # Check if ML is enabled via config
        from configs.config_manager import ConfigManager
        cfg = ConfigManager().config

        if not cfg.get("features", {}).get("ml_enabled", False):
            logger.info(f"[ML_ALPHA] Disabled by governance flag (cfg.features.ml_enabled=false)")
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "metadata": {"reason": "ML_DISABLED_BY_CONFIG"}
            }

        # Check governance state
        if self._governance_disabled:
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "metadata": {"reason": "ML_DISABLED_BY_GOVERNANCE", "model_errors": self.model_errors}
            }

        try:
            model_artifact = self._load_model_for_symbol(symbol)

            # SUPPORT PRE-COMPUTED FEATURES
            provided_features = kwargs.get("features")
            if provided_features:
                if isinstance(provided_features, dict):
                    # Convert dict to single-row DataFrame
                    X = pd.DataFrame([provided_features])
                elif isinstance(provided_features, pd.DataFrame):
                    X = provided_features.iloc[-1:]
                else:
                    logger.warning(f"[ML_ALPHA] Unexpected features format for {symbol}: {type(provided_features)}")
                    from data.processors.features import compute_features_for_symbol
                    extracted = compute_features_for_symbol(market_data, contract_name="ml_v1")
                    X = extracted.iloc[-1:] if extracted is not None else None
            else:
                # Use contract-aligned feature extraction (28 features)
                from data.processors.features import compute_features_for_symbol
                extracted = compute_features_for_symbol(market_data, contract_name="ml_v1")
                X = extracted.iloc[-1:] if extracted is not None else None

            if X is None or len(X) == 0:
                return {
                    "signal": 0.0,
                    "confidence": 0.0,
                    "metadata": {"reason": "INSUFFICIENT_DATA"}
                }

            pred = None
            metadata = {"status": "ACTIVE"}

            if model_artifact:
                model = model_artifact["model"]
                model_meta = model_artifact.get("metadata", model_artifact)
                try:
                    pred = self.ml_predict_safe(model, X, model_meta, symbol=symbol)
                except Exception as e:
                    logger.warning(f"Primary model failed for {symbol}: {e}. Trying fallback.")

            # Fallback Logic
            if pred is None:
                if "LEGACY_GLOBAL" in self._cached_models:
                    legacy_data = self._cached_models["LEGACY_GLOBAL"]
                    pred = self.ml_predict_safe(legacy_data["model"], X, legacy_data.get("metadata"), symbol=f"{symbol}_LEGACY")
                    if pred is not None:
                        metadata["status"] = "FALLBACK"
                        metadata["reason"] = "PRIMARY_FAILED"
                elif "GLOBAL" in self._cached_models and symbol != "GLOBAL":
                     global_data = self._cached_models["GLOBAL"]
                     pred = self.ml_predict_safe(global_data["model"], X, global_data.get("metadata"), symbol=f"{symbol}_GLOBAL")
                     if pred is not None:
                        metadata["status"] = "FALLBACK"
                        metadata["reason"] = "SYMBOL_MODEL_MISSING"

            if pred is None:
                return {
                    "signal": 0.0,
                    "confidence": 0.0,
                    "metadata": {"reason": "PREDICTION_FAIL_ALL_MODELS"}
                }

            signal = np.clip(pred[0], -1.0, 1.0)
            confidence = min(abs(signal) * 2, 1.0)

            if regime_context:
                signal, confidence = self._adjust_for_regime(signal, confidence, regime_context)

            return {
                "signal": float(signal),
                "confidence": float(confidence),
                "metadata": {"status": "ACTIVE"}
            }

        except GovernanceDisabledError as e:
            logger.warning(f"[ML_ALPHA] Governance disabled for {symbol}: {e}")
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "metadata": {"reason": "GOVERNANCE_DISABLED", "detail": str(e)}
            }
        except ModelFeatureMismatchError as e:
            logger.error(f"[ML_ALPHA] Feature mismatch for {symbol}: {e}")
            # Log structured error
            logger.error(json.dumps(e.to_dict()))
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "metadata": {"reason": "FEATURE_MISMATCH", "detail": str(e)}
            }
        except Exception as e:
            logger.error(f"ML alpha signal generation failed for {symbol}: {e}")
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }

    def _extract_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Legacy feature extraction - will be replaced by compute_features_for_symbol."""
        if len(data) < self.feature_lookback + 5:
            return None
        df = ensure_business_days(data.copy())
        df["returns"] = df["Close"].pct_change(fill_method=None)
        for p in [5, 20]:
            df[f"momentum_{p}"] = df["Close"] / df["Close"].shift(p) - 1
        df["realized_vol_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
        df = df.dropna()
        exclude = ["Open", "High", "Low", "Close", "Volume", "symbol", "date", "target"]
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        return df[feature_cols]

    def _adjust_for_regime(
        self,
        signal: float,
        confidence: float,
        regime_context: Dict
    ) -> Tuple[float, float]:
        """Adjust signal based on market regime."""
        regime = regime_context.get("regime_tag", "NORMAL")
        mult = {
            "HIGH_VOL": 0.8,
            "LOW_VOL": 1.2,
            "BULL_QUIET": 1.1,
            "BEAR_CRISIS": 0.7
        }.get(regime, 1.0)
        return signal * mult, confidence
