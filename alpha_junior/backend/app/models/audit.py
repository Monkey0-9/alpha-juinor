"""
Audit Log model
Immutable audit trail of all significant actions
Required for compliance, security forensics, regulatory reporting
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, INET
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class AuditAction(str, Enum):
    """Types of auditable actions"""
    # User actions
    USER_REGISTERED = "user_registered"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_UPDATED = "user_updated"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"
    TWO_FA_ENABLED = "two_fa_enabled"
    TWO_FA_DISABLED = "two_fa_disabled"
    EMAIL_VERIFIED = "email_verified"
    
    # Profile actions
    PROFILE_UPDATED = "profile_updated"
    KYC_SUBMITTED = "kyc_submitted"
    KYC_APPROVED = "kyc_approved"
    KYC_REJECTED = "kyc_rejected"
    ACCREDITATION_ATTESTED = "accreditation_attested"
    
    # Fund actions (manager)
    FUND_CREATED = "fund_created"
    FUND_UPDATED = "fund_updated"
    FUND_PUBLISHED = "fund_published"
    FUND_CLOSED = "fund_closed"
    FUND_DELETED = "fund_deleted"
    FUND_NAV_UPDATED = "fund_nav_updated"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"
    
    # Investment actions
    INVESTMENT_SUBMITTED = "investment_submitted"
    INVESTMENT_APPROVED = "investment_approved"
    INVESTMENT_REJECTED = "investment_rejected"
    INVESTMENT_ACTIVATED = "investment_activated"
    INVESTMENT_CANCELLED = "investment_cancelled"
    REDEMPTION_REQUESTED = "redemption_requested"
    REDEMPTION_PROCESSED = "redemption_processed"
    AGREEMENT_SIGNED = "agreement_signed"
    
    # Capital / Distribution
    CAPITAL_CALL_CREATED = "capital_call_created"
    CAPITAL_CALL_UPDATED = "capital_call_updated"
    DISTRIBUTION_CREATED = "distribution_created"
    
    # Administrative
    USER_ROLE_CHANGED = "user_role_changed"
    USER_DEACTIVATED = "user_deactivated"
    FUND_STATUS_CHANGED = "fund_status_changed"
    KYC_STATUS_CHANGED = "kyc_status_changed"
    
    # Security
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    KILL_SWITCH_TRIGGERED = "kill_switch_triggered"


class AuditLog(Base):
    """
    Immutable audit log entry
    Every state-changing action should create an audit log entry
    """
    
    __tablename__ = "audit_logs"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Actor (who performed the action)
    actor_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=True,  # Can be null for system actions or unauthenticated attempts
        index=True
    )
    
    # Action details
    action: Mapped[AuditAction] = mapped_column(String(50), nullable=False, index=True)
    
    # Target (what was affected)
    target_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True  # e.g., "user", "fund", "investment"
    )
    target_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True, index=True)
    
    # Request context
    ip_address: Mapped[Optional[str]] = mapped_column(INET, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    
    # Full payload (before/after for updates)
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    # Example structure:
    # {
    #   "before": {"status": "pending", "nav": 100.0},
    #   "after": {"status": "active", "nav": 105.5},
    #   "metadata": {"reason": "manual approval"}
    # }
    
    # Sensitive operations may require additional verification
    requires_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reviewed_by_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=True
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamp (indexed for time-range queries)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    # Relationships
    actor: Mapped[Optional["User"]] = relationship("User", foreign_keys=[actor_id], lazy="selectin")
    reviewed_by: Mapped[Optional["User"]] = relationship("User", foreign_keys=[reviewed_by_id], lazy="selectin")
    
    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action={self.action}, actor={self.actor_id}, target={self.target_type}:{self.target_id})>"
    
    @property
    def is_system_action(self) -> bool:
        """Check if this was a system/automated action"""
        return self.actor_id is None
    
    @property
    def has_changes(self) -> bool:
        """Check if this audit log contains before/after changes"""
        if self.payload:
            return "before" in self.payload and "after" in self.payload
        return False
