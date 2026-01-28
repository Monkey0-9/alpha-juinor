"""
contracts/alpha_contract.py

Canonical Alpha Contract (Ticket 5)

Enforces standard output schema for ALL alpha signals.
Every alpha must return distributional outputs in CANONICAL UNITS:
- mu: Daily expected return in decimal (e.g., 0.01 = 1%)
- sigma: Daily volatility in decimal
- cvar_95: 95% CVaR as negative decimal (loss)
- confidence: Signal confidence [0, 1]

Plus metadata for audit:
- provider: Alpha source identifier
- model_version: Semantic version
- input_schema_hash: MD5 of input features
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

logger = logging.getLogger("ALPHA_CONTRACT")


class AlphaDecision(str, Enum):
    """High-level alpha decision."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ABSTAIN = "ABSTAIN"  # Model explicitly declines to provide signal


@dataclass
class AlphaOutput:
    """
    Canonical Alpha Output Contract.

    ALL alphas MUST output in these units:
    - mu: Daily expected return (decimal, e.g., 0.01 = 1% daily)
    - sigma: Daily volatility (decimal, positive)
    - cvar_95: 95% Conditional VaR (negative decimal, e.g., -0.05 = 5% daily loss)
    - confidence: Signal confidence [0, 1]

    Metadata (required for audit):
    - provider: Alpha source (e.g., "ml_v2", "arima", "momentum")
    - model_version: Semantic version (e.g., "2.1.0")
    - input_schema_hash: MD5 of input features used
    """
    # Core distributional outputs
    mu: float                    # Daily expected return (decimal)
    sigma: float                 # Daily volatility (decimal)
    cvar_95: float               # 95% CVaR (negative decimal)
    confidence: float            # [0, 1]

    # Metadata for audit
    provider: str                # Alpha source identifier
    model_version: str           # Semantic version
    input_schema_hash: str       # MD5 of input features

    # Optional enrichment
    p_loss: float = 0.5          # Probability of loss [0, 1]
    distribution_type: str = "NORMAL" # NORMAL, T_DIST, EMPIRICAL
    model_disagreement: float = 0.0   # 0.0 = Consensus, 1.0 = Chaos
    decision: AlphaDecision = AlphaDecision.HOLD
    explanation: str = ""        # Human-readable rationale

    # Timestamps
    generated_at: str = ""       # ISO timestamp

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat() + 'Z'

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate contract compliance.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # ===== RANGE CHECKS (STRICT) =====

        # mu: Daily return typically -5% to +5%
        if not (-0.10 <= self.mu <= 0.10):
            errors.append(f"mu out of valid range [-0.10, 0.10]: {self.mu}")

        # sigma: Daily vol typically 0.1% to 15%
        if not (0.0001 <= self.sigma <= 0.20):
            errors.append(f"sigma out of valid range [0.0001, 0.20]: {self.sigma}")

        # cvar_95: Must be negative (it's a loss measure)
        if not (-0.30 <= self.cvar_95 < 0):
            errors.append(f"cvar_95 out of valid range [-0.30, 0): {self.cvar_95}")

        # confidence: Must be [0, 1]
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence out of valid range [0, 1]: {self.confidence}")

        # p_loss: Must be [0, 1]
        if not (0.0 <= self.p_loss <= 1.0):
            errors.append(f"p_loss out of valid range [0, 1]: {self.p_loss}")

        # model_disagreement: Must be [0, 1]
        if not (0.0 <= self.model_disagreement <= 1.0):
             errors.append(f"model_disagreement out of valid range [0, 1]: {self.model_disagreement}")

        # distribution_type: Check valid
        if self.distribution_type not in ["NORMAL", "T_DIST", "EMPIRICAL"]:
             errors.append(f"distribution_type invalid: {self.distribution_type}")

        # ===== CONSISTENCY CHECKS =====

        # sigma must be positive
        if self.sigma <= 0:
            errors.append(f"sigma must be positive: {self.sigma}")

        # cvar should be more negative than mu for losses
        if self.mu < 0 and self.cvar_95 > self.mu:
            errors.append(f"cvar_95 ({self.cvar_95}) should be <= mu ({self.mu}) for negative returns")

        # ===== METADATA CHECKS =====

        if not self.provider:
            errors.append("provider is required")

        if not self.model_version:
            errors.append("model_version is required")

        if not self.input_schema_hash:
            errors.append("input_schema_hash is required")

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(json.dumps({
                "event": "ALPHA_CONTRACT_VIOLATION",
                "provider": self.provider,
                "model_version": self.model_version,
                "errors": errors,
                "mu": self.mu,
                "sigma": self.sigma,
                "cvar_95": self.cvar_95
            }))

        return is_valid, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['decision'] = self.decision.value if isinstance(self.decision, AlphaDecision) else self.decision
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlphaOutput":
        """Create from dictionary."""
        decision = d.get('decision', 'HOLD')
        if isinstance(decision, str):
            try:
                decision = AlphaDecision(decision)
            except ValueError:
                decision = AlphaDecision.HOLD

        return cls(
            mu=float(d.get('mu', 0.0)),
            sigma=float(d.get('sigma', 0.01)),
            cvar_95=float(d.get('cvar_95', -0.01)),
            confidence=float(d.get('confidence', 0.5)),
            provider=d.get('provider', 'unknown'),
            model_version=d.get('model_version', 'v0.0.0'),
            input_schema_hash=d.get('input_schema_hash', ''),
            p_loss=float(d.get('p_loss', 0.5)),
            distribution_type=d.get('distribution_type', 'NORMAL'),
            model_disagreement=float(d.get('model_disagreement', 0.0)),
            decision=decision,
            explanation=d.get('explanation', ''),
            generated_at=d.get('generated_at', '')
        )

    @classmethod
    def abstain(cls, provider: str, reason: str) -> "AlphaOutput":
        """
        Create an ABSTAIN signal (model declines to predict).

        This is different from HOLD - abstain means "I don't know"
        and should not count toward ensemble votes.
        """
        return cls(
            mu=0.0,
            sigma=0.02,  # Neutral vol
            cvar_95=-0.02,
            confidence=0.0,  # Zero confidence = abstain
            provider=provider,
            model_version="abstain",
            input_schema_hash="",
            p_loss=0.5,
            decision=AlphaDecision.ABSTAIN,
            explanation=reason
        )

    def derive_decision(self, mu_threshold: float = 0.002) -> AlphaDecision:
        """
        Derive trading decision from distributional output.

        Args:
            mu_threshold: Minimum absolute mu to trigger BUY/SELL
        """
        if self.confidence < 0.3:
            return AlphaDecision.ABSTAIN

        if self.mu > mu_threshold:
            return AlphaDecision.BUY
        elif self.mu < -mu_threshold:
            return AlphaDecision.SELL
        else:
            return AlphaDecision.HOLD


class AlphaContractEnforcer:
    """
    Central enforcement of alpha contract.

    All alpha outputs MUST pass through this enforcer before
    reaching the PM Brain or execution layer.

    This enforcer:
    1. Validates outputs against the contract
    2. Normalizes units if needed
    3. Rejects invalid outputs with audit trail
    4. Enriches with metadata
    """

    # Clipping ranges for normalization
    MU_RANGE = (-0.05, 0.05)         # ±5% daily
    SIGMA_RANGE = (0.001, 0.10)      # 0.1% to 10% daily vol
    CVAR_RANGE = (-0.20, -0.001)     # Max 20% daily loss

    def __init__(self):
        self._violation_count = 0
        self._total_processed = 0

    def normalize_and_validate(
        self,
        raw_output: Dict[str, Any],
        provider: str,
        model_version: str,
        input_features: List[str] = None,
        current_price: float = None
    ) -> Tuple[Optional[AlphaOutput], List[str]]:
        """
        Normalize and validate alpha output.

        Args:
            raw_output: Raw dictionary output from alpha model
            provider: Alpha source identifier
            model_version: Model version
            input_features: List of input feature names (for hash)
            current_price: Current asset price (for price-delta normalization)

        Returns:
            Tuple of (AlphaOutput or None, list of errors/warnings)
        """
        self._total_processed += 1
        warnings = []

        # Compute schema hash
        if input_features:
            schema_hash = hashlib.md5(json.dumps(sorted(input_features)).encode()).hexdigest()[:16]
        else:
            schema_hash = raw_output.get('input_schema_hash', 'unknown')

        # Extract raw values
        mu = float(raw_output.get('mu', 0.0))
        sigma = float(raw_output.get('sigma', 0.02))
        cvar_95 = float(raw_output.get('cvar_95', -0.02))
        confidence = float(raw_output.get('confidence', 0.5))
        model_disagreement = float(raw_output.get('model_disagreement', 0.0))
        dist_type = raw_output.get('distribution_type', 'NORMAL')

        # ===== PRICE-DELTA NORMALIZATION =====
        # If mu looks like a price delta (abs > 1.0), convert to return
        if abs(mu) > 1.0 and current_price and current_price > 0:
            original_mu = mu
            mu = mu / current_price
            warnings.append(f"Normalized mu from price-delta {original_mu:.4f} to return {mu:.6f}")

        # ===== HARD CLIPPING =====

        # Clip mu
        if mu < self.MU_RANGE[0] or mu > self.MU_RANGE[1]:
            original_mu = mu
            mu = max(self.MU_RANGE[0], min(self.MU_RANGE[1], mu))
            warnings.append(f"Clipped mu from {original_mu:.6f} to {mu:.6f}")

        # Clip sigma
        if sigma < self.SIGMA_RANGE[0] or sigma > self.SIGMA_RANGE[1]:
            original_sigma = sigma
            sigma = max(self.SIGMA_RANGE[0], min(self.SIGMA_RANGE[1], sigma))
            warnings.append(f"Clipped sigma from {original_sigma:.6f} to {sigma:.6f}")

        # Clip cvar
        if cvar_95 < self.CVAR_RANGE[0] or cvar_95 > self.CVAR_RANGE[1]:
            original_cvar = cvar_95
            cvar_95 = max(self.CVAR_RANGE[0], min(self.CVAR_RANGE[1], cvar_95))
            warnings.append(f"Clipped cvar_95 from {original_cvar:.6f} to {cvar_95:.6f}")

        # Ensure cvar is negative
        if cvar_95 >= 0:
            cvar_95 = -0.001
            warnings.append("cvar_95 was non-negative, set to -0.001")

        # Clip confidence
        confidence = max(0.0, min(1.0, confidence))

        # Clip model disagreement
        model_disagreement = max(0.0, min(1.0, model_disagreement))

        # ===== CONSISTENCY ENFORCEMENT =====

        # Ensure cvar <= mu for coherence
        if cvar_95 > mu:
            cvar_95 = mu - 0.01
            cvar_95 = max(self.CVAR_RANGE[0], cvar_95)
            warnings.append(f"Adjusted cvar_95 to ensure cvar_95 <= mu")

        # Compute p_loss
        p_loss = raw_output.get('p_loss', 0.5)
        if mu < 0:
            p_loss = min(0.9, max(p_loss, 0.5))  # Higher p_loss for negative mu
        elif mu > 0:
            p_loss = max(0.1, min(p_loss, 0.5))  # Lower p_loss for positive mu

        # ===== CREATE OUTPUT =====

        alpha_output = AlphaOutput(
            mu=round(mu, 6),
            sigma=round(sigma, 6),
            cvar_95=round(cvar_95, 6),
            confidence=round(confidence, 4),
            provider=provider,
            model_version=model_version,
            input_schema_hash=schema_hash,
            p_loss=round(p_loss, 4),
            distribution_type=dist_type,
            model_disagreement=round(model_disagreement, 4),
            explanation=raw_output.get('explanation', '')
        )

        # Derive decision
        alpha_output.decision = alpha_output.derive_decision()

        # Final validation
        is_valid, errors = alpha_output.validate()

        if not is_valid:
            self._violation_count += 1
            logger.error(json.dumps({
                "event": "ALPHA_CONTRACT_REJECTED",
                "provider": provider,
                "model_version": model_version,
                "errors": errors,
                "warnings": warnings
            }))
            return None, errors

        if warnings:
            logger.info(json.dumps({
                "event": "ALPHA_NORMALIZED",
                "provider": provider,
                "warnings": warnings
            }))

        return alpha_output, warnings

    def get_stats(self) -> Dict[str, Any]:
        """Get enforcement statistics."""
        return {
            "total_processed": self._total_processed,
            "violation_count": self._violation_count,
            "violation_rate": self._violation_count / max(1, self._total_processed)
        }


# Singleton instance
_enforcer_instance: Optional[AlphaContractEnforcer] = None


def get_alpha_enforcer() -> AlphaContractEnforcer:
    """Get singleton AlphaContractEnforcer instance."""
    global _enforcer_instance
    if _enforcer_instance is None:
        _enforcer_instance = AlphaContractEnforcer()
    return _enforcer_instance


def validate_alpha_output(output: Dict[str, Any]) -> Optional[AlphaOutput]:
    """
    Legacy compatibility wrapper.

    Use AlphaContractEnforcer.normalize_and_validate for full functionality.
    """
    enforcer = get_alpha_enforcer()
    result, errors = enforcer.normalize_and_validate(
        raw_output=output,
        provider=output.get('provider', 'unknown'),
        model_version=output.get('model_version', 'v0.0.0')
    )
    return result
