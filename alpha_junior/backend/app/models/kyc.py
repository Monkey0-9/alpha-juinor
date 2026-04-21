"""
KYC Submission model
Know Your Customer / Know Your Business verification
"""

from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class IDType(str, Enum):
    """Types of identification documents"""
    PASSPORT = "passport"
    NATIONAL_ID = "national_id"
    DRIVERS_LICENSE = "drivers_license"
    RESIDENCE_PERMIT = "residence_permit"
    COMPANY_REGISTRATION = "company_registration"


class KYCStatus(str, Enum):
    """KYC verification status"""
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ADDITIONAL_INFO_REQUIRED = "additional_info_required"


class KYCSubmission(Base, TimestampMixin):
    """
    KYC document submission for identity verification
    Supports both individual and institutional investors
    """
    
    __tablename__ = "kyc_submissions"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # ID document details
    id_type: Mapped[IDType] = mapped_column(String(50), nullable=False)
    id_number: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Encrypted
    id_country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # S3 document storage
    id_front_s3_key: Mapped[str] = mapped_column(String(500), nullable=False)
    id_front_s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    
    id_back_s3_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    id_back_s3_bucket: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    selfie_s3_key: Mapped[str] = mapped_column(String(500), nullable=False)
    selfie_s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    
    proof_of_address_s3_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    proof_of_address_s3_bucket: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Additional documents for institutions
    certificate_of_incorporation_s3_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    articles_of_association_s3_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    beneficial_ownership_s3_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Status tracking
    status: Mapped[KYCStatus] = mapped_column(
        String(50),
        default=KYCStatus.PENDING,
        nullable=False,
        index=True
    )
    
    # Submission timeline
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Review details
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    reviewer_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=True
    )
    
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    additional_info_requested: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Third-party verification (if integrated)
    verification_provider: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    verification_ref_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    verification_result: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # IP and metadata for audit
    submitted_from_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Document verification scores (if using AI verification)
    id_match_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)
    liveness_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="kyc_submissions", foreign_keys=[user_id])
    reviewer: Mapped[Optional["User"]] = relationship("User", foreign_keys=[reviewer_id])
    
    def __repr__(self) -> str:
        return f"<KYCSubmission(id={self.id}, user={self.user_id}, status={self.status})>"
    
    @property
    def is_pending(self) -> bool:
        return self.status in [KYCStatus.PENDING, KYCStatus.UNDER_REVIEW, KYCStatus.ADDITIONAL_INFO_REQUIRED]
    
    @property
    def is_approved(self) -> bool:
        return self.status == KYCStatus.APPROVED
