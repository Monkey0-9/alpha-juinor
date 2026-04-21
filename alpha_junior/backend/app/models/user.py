"""
User and UserProfile models
Authentication, roles, KYC status, 2FA
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, Numeric, Index, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin, MoneyType

if TYPE_CHECKING:
    from app.models.fund import Fund
    from app.models.investment import Investment
    from app.models.kyc import KYCSubmission
    from app.models.notifications import Notification
    from app.models.watchlist import Watchlist
    from app.models.messages import Message


class UserRole(str, Enum):
    """User roles for RBAC"""
    INVESTOR = "investor"
    MANAGER = "manager"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


class KYCStatus(str, Enum):
    """KYC verification status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class AccreditationStatus(str, Enum):
    """Investor accreditation status"""
    RETAIL = "retail"
    ACCREDITED = "accredited"
    INSTITUTIONAL = "institutional"


class User(Base, TimestampMixin, SoftDeleteMixin):
    """
    Core user account model
    Authentication, roles, KYC status, 2FA
    """
    
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Authentication
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Role-based access control
    role: Mapped[UserRole] = mapped_column(
        String(50),
        default=UserRole.INVESTOR,
        nullable=False,
        index=True
    )
    
    # KYC/Compliance
    kyc_status: Mapped[KYCStatus] = mapped_column(
        String(50),
        default=KYCStatus.PENDING,
        nullable=False,
        index=True
    )
    
    # 2FA
    is_2fa_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    totp_secret: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Session management
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    
    # Email verification
    email_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    email_verification_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    
    # Password reset
    password_reset_token: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    password_reset_expires: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    profile: Mapped[Optional["UserProfile"]] = relationship(
        "UserProfile",
        back_populates="user",
        uselist=False,
        lazy="joined"
    )
    
    managed_funds: Mapped[List["Fund"]] = relationship(
        "Fund",
        back_populates="manager",
        lazy="selectin"
    )
    
    investments: Mapped[List["Investment"]] = relationship(
        "Investment",
        back_populates="investor",
        lazy="selectin"
    )
    
    kyc_submissions: Mapped[List["KYCSubmission"]] = relationship(
        "KYCSubmission",
        back_populates="user",
        lazy="selectin"
    )
    
    notifications: Mapped[List["Notification"]] = relationship(
        "Notification",
        back_populates="user",
        lazy="selectin",
        order_by="desc(Notification.created_at)"
    )
    
    watchlists: Mapped[List["Watchlist"]] = relationship(
        "Watchlist",
        back_populates="investor",
        lazy="selectin"
    )
    
    sent_messages: Mapped[List["Message"]] = relationship(
        "Message",
        foreign_keys="Message.sender_id",
        back_populates="sender",
        lazy="selectin"
    )
    
    received_messages: Mapped[List["Message"]] = relationship(
        "Message",
        foreign_keys="Message.recipient_id",
        back_populates="recipient",
        lazy="selectin"
    )
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_users_role_kyc', 'role', 'kyc_status'),
        Index('ix_users_active_email', 'is_active', 'email'),
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"
    
    @property
    def is_manager(self) -> bool:
        return self.role in [UserRole.MANAGER, UserRole.ADMIN, UserRole.SUPERADMIN]
    
    @property
    def is_admin(self) -> bool:
        return self.role in [UserRole.ADMIN, UserRole.SUPERADMIN]
    
    @property
    def is_superadmin(self) -> bool:
        return self.role == UserRole.SUPERADMIN
    
    @property
    def can_invest(self) -> bool:
        """Check if user can make investments"""
        return (
            self.is_active 
            and not self.is_deleted 
            and self.kyc_status == KYCStatus.APPROVED
            and self.email_verified_at is not None
        )


class UserProfile(Base):
    """
    Extended user profile information
    Investor details, accreditation, social links
    """
    
    __tablename__ = "user_profiles"
    
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )
    
    # Personal info
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Accreditation
    accreditation_status: Mapped[AccreditationStatus] = mapped_column(
        String(50),
        default=AccreditationStatus.RETAIL,
        nullable=False
    )
    
    # Financial profile (for suitability)
    net_worth_bracket: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True  # e.g., "100k-500k", "500k-1m", "1m-5m", "5m+"
    )
    annual_income_bracket: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True
    )
    investment_experience: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True  # "none", "limited", "moderate", "extensive"
    )
    
    # Social/Professional
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    bio: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    linkedin_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    company_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    job_title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Bank details (encrypted at application level)
    bank_account_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bank_routing_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="profile")
    
    def __repr__(self) -> str:
        return f"<UserProfile(user_id={self.user_id}, name={self.full_name})>"
