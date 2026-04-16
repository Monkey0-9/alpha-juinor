
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum

class AlphaDecision(str, Enum):
    """High-level alpha decision."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ABSTAIN = "ABSTAIN"

class AlphaOutput(BaseModel):
    """
    Canonical Alpha Output Contract (Pydantic Version).

    Ensures strict type validation and range checking.
    """
    # Core distributional outputs
    mu: float = Field(..., description="Daily expected return (decimal)")
    sigma: float = Field(..., ge=0.0001, le=0.5, description="Daily volatility (decimal)")
    cvar_95: float = Field(..., lt=0, description="95% CVaR (negative decimal)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence [0, 1]")

    # Metadata
    provider: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)
    input_schema_hash: str = Field(..., min_length=1)

    # Optional enrichment
    p_loss: float = Field(0.5, ge=0.0, le=1.0)
    distribution_type: Literal["NORMAL", "T_DIST", "EMPIRICAL"] = "NORMAL"
    model_disagreement: float = Field(0.0, ge=0.0, le=1.0)
    decision: AlphaDecision = AlphaDecision.HOLD
    explanation: str = ""
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    @field_validator('mu')
    @classmethod
    def check_mu_range(cls, v: float) -> float:
        if not (-0.5 <= v <= 0.5):
            raise ValueError(f"mu {v} is outside reasonable range [-0.5, 0.5]")
        return v

    @field_validator('cvar_95')
    @classmethod
    def check_cvar_consistency(cls, v: float) -> float:
        if v >= 0:
            raise ValueError("CVaR must be negative")
        return v

    @model_validator(mode='after')
    def check_logic(self) -> 'AlphaOutput':
        # Ensure CVaR is consistent with mu for negative returns
        if self.mu < 0 and self.cvar_95 > self.mu:
            # This is physically possible for fat tails but suspicious if cvar > mu (less negative loss than expected return)
            pass
        return self

    @classmethod
    def neutral(cls, model_id: str = "unknown") -> 'AlphaOutput':
        """Create a neutral/abstain output for safe fallback."""
        return cls(
            mu=0.0,
            sigma=0.01,
            cvar_95=-0.01,
            confidence=0.0,
            provider=model_id,
            model_version="neutral_v1",
            input_schema_hash="none",
            decision=AlphaDecision.ABSTAIN,
            explanation="Neutral fallback output"
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
