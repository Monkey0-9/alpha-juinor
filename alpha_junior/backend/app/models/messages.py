"""
Message model
Investor ↔ Manager communication system
Threaded conversations with fund context
"""

from datetime import datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.fund import Fund


class MessageStatus(str, Enum):
    """Message delivery status"""
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"


class Message(Base, TimestampMixin):
    """
    Message between users (investor ↔ manager)
    Supports threaded conversations with fund context
    """
    
    __tablename__ = "messages"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Sender and recipient
    sender_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )
    
    recipient_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )
    
    # Optional fund context
    fund_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("funds.id"),
        nullable=True,
        index=True
    )
    
    # Threading
    thread_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("messages.id"),
        nullable=True,
        index=True
    )
    
    # Message content
    subject: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Status
    status: Mapped[MessageStatus] = mapped_column(
        String(50),
        default=MessageStatus.SENT,
        nullable=False
    )
    
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Sent timestamp
    sent_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Notification tracking
    notification_sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notification_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Email tracking (if sent via email)
    email_sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    email_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Reply tracking
    reply_to_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("messages.id"),
        nullable=True
    )
    
    # Internal notes (for manager/admin use)
    internal_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_internal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Attachments (JSON array of S3 keys)
    attachments: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    
    # Relationships
    sender: Mapped["User"] = relationship(
        "User",
        foreign_keys=[sender_id],
        back_populates="sent_messages",
        lazy="joined"
    )
    
    recipient: Mapped["User"] = relationship(
        "User",
        foreign_keys=[recipient_id],
        back_populates="received_messages",
        lazy="joined"
    )
    
    fund: Mapped[Optional["Fund"]] = relationship("Fund", back_populates="messages")
    
    # Thread relationships
    parent_thread: Mapped[Optional["Message"]] = relationship(
        "Message",
        remote_side=[id],
        foreign_keys=[thread_id],
        lazy="selectin"
    )
    
    replies: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="parent_thread",
        lazy="selectin"
    )
    
    # Reply to specific message
    reply_to: Mapped[Optional["Message"]] = relationship(
        "Message",
        remote_side=[id],
        foreign_keys=[reply_to_id],
        lazy="joined"
    )
    
    # Indexes for conversation queries
    __table_args__ = (
        Index('ix_messages_conversation', 'sender_id', 'recipient_id', 'sent_at'),
        Index('ix_messages_thread', 'thread_id', 'sent_at'),
        Index('ix_messages_fund_recipient', 'fund_id', 'recipient_id', 'is_read'),
    )
    
    def __repr__(self) -> str:
        return f"<Message(id={self.id}, from={self.sender_id}, to={self.recipient_id}, subject={self.subject})>"
    
    def mark_as_read(self) -> None:
        """Mark message as read"""
        self.is_read = True
        self.read_at = datetime.utcnow()
        self.status = MessageStatus.READ
    
    @property
    def is_thread_starter(self) -> bool:
        """Check if this message starts a thread"""
        return self.thread_id is None or self.thread_id == self.id
    
    @property
    def preview(self) -> str:
        """Short preview of message body"""
        return self.body[:100] + "..." if len(self.body) > 100 else self.body
