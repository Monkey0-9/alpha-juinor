"""
contracts/alpha_output.py

Alpha Distributional Contract - enforces standard output schema.
All alpha families MUST return distributional outputs.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class AlphaDistribution:
    """
    Mandatory output contract for all alpha families.

    Attributes:
        mu: Expected return (daily)
        sigma: Volatility (daily)
        p_loss: Probability of loss [0,1]
        cvar_95: Conditional Value at Risk at 95% confidence
        confidence: Signal confidence [0,1]
    """
    mu: float
    sigma: float
    p_loss: float
    cvar_95: float
    confidence: float
    distribution_type: str = "NORMAL"

    def validate(self) -> bool:
        """Validate contract compliance"""
        errors = []

        # Range checks
        if not (-1.0 <= self.mu <= 1.0):
            errors.append(f"mu out of range: {self.mu}")

        if not (0.0 <= self.sigma <= 1.0):
            errors.append(f"sigma out of range: {self.sigma}")

        if not (0.0 <= self.p_loss <= 1.0):
            errors.append(f"p_loss out of range: {self.p_loss}")

        if not (-1.0 <= self.cvar_95 <= 0.0):
            errors.append(f"cvar_95 out of range: {self.cvar_95}")

        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"confidence out of range: {self.confidence}")

        # Logical consistency
        if self.sigma <= 0:
            errors.append("sigma must be positive")

        if errors:
            logger.error(f"Alpha contract validation failed: {errors}")
            return False

        return True

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "p_loss": self.p_loss,
            "cvar_95": self.cvar_95,
            "confidence": self.confidence
        }

def validate_alpha_output(output: dict) -> Optional[AlphaDistribution]:
    """
    Validate and convert alpha output to contract.

    Returns AlphaDistribution if valid, None if invalid.
    """
    required_fields = ["mu", "sigma", "p_loss", "cvar_95", "confidence"]

    # Check for missing fields
    missing = [f for f in required_fields if f not in output]
    if missing:
        logger.error(f"Alpha output missing required fields: {missing}")
        return None

    try:
        dist = AlphaDistribution(
            mu=float(output["mu"]),
            sigma=float(output["sigma"]),
            p_loss=float(output["p_loss"]),
            cvar_95=float(output["cvar_95"]),
            confidence=float(output["confidence"])
        )

        if dist.validate():
            return dist
        else:
            return None
    except Exception as e:
        logger.error(f"Failed to parse alpha output: {e}")
        return None
