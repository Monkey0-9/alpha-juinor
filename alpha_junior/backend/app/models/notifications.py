"""
Notification model
User notifications via WebSocket, email, in-app
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class NotificationType(str, Enum):
    """Types of notifications"""
    # Investment related
    INVESTMENT_APPROVED = "investment_approved"
    INVESTMENT_REJECTED = "investment_rejected"
    INVESTMENT_REDEEMED = "investment_redeemed"
    
    # Fund related
    FUND_UPDATE = "fund_update"
    FUND_NAV_UPDATE = "fund_nav_update"
    CAPITAL_CALL = "capital_call"
    DISTRIBUTION = "distribution"
    NEW_DOCUMENT = "new_document"
    REPORT_AVAILABLE = "report_available"
    
    # KYC related
    KYC_APPROVED = "kyc_approved"
    KYC_REJECTED = "kyc_rejected"
    KYC_ADDITIONAL_INFO = "kyc_additional_info"
    
    # Account related
    PASSWORD_CHANGED = "password_changed"
    LOGIN_NEW_DEVICE = "login_new_device"
    
    # Message related
    NEW_MESSAGE = "new_message"
    
    # System
    SYSTEM_MAINTENANCE = "system_maintenance"
    FEATURE_UPDATE = "feature_update"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationChannel(str, Enum):
    """How notification was delivered"""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBSOCKET = "websocket"


class Notification(Base, TimestampMixin):
    """
    User notification
    Multi-channel delivery tracking
    """
    
    __tablename__ = "notifications"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Notification content
    type: Mapped[NotificationType] = mapped_column(String(50), nullable=False, index=True)
    priority: Mapped[NotificationPriority] = mapped_column(
        String(50),
        default=NotificationPriority.NORMAL,
        nullable=False
    )
    
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Action / CTA
    action_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    action_text: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Related entity metadata
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    # Example: {"investment_id": 123, "fund_id": 456, "amount": 10000}
    
    # Read status
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Delivery tracking
    delivered_via: Mapped[Optional[NotificationChannel]] = mapped_column(String(50), nullable=True)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    email_sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    email_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    websocket_pushed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    websocket_pushed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="notifications")
    
    def __repr__(self) -> str:
        return f"<Notification(id={self.id}, user={self.user_id}, type={self.type}, read={self.is_read})>"
    
    def mark_as_read(self) -> None:
        """Mark notification as read"""
        self.is_read = True
        self.read_at = datetime.utcnow()
