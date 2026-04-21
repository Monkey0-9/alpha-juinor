"""
Investment, CapitalCall, Distribution models
Investment lifecycle and capital movements
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, String, Text, Numeric, Index, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin, MoneyType

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.fund import Fund


class InvestmentStatus(str, Enum):
    """Investment lifecycle status"""
    PENDING = "pending"           # Awaiting manager approval
    APPROVED = "approved"         # Approved but not yet active
    ACTIVE = "active"             # Investment is live
    REDEEMING = "redeeming"       # Redemption requested
    REDEEMED = "redeemed"         # Fully redeemed
    CANCELLED = "cancelled"       # Cancelled by manager or user
    REJECTED = "rejected"         # Manager rejected


class Investment(Base, TimestampMixin, SoftDeleteMixin):
    """
    Investor's investment in a fund
    Tracks subscription, units, NAV entry, current value
    """
    
    __tablename__ = "investments"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Relationships
    investor_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )
    
    fund_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("funds.id"),
        nullable=False,
        index=True
    )
    
    # Investment details
    amount: Mapped[Decimal] = mapped_column(MoneyType, nullable=False)  # Original investment amount
    currency: Mapped[str] = mapped_column(String(10), default="USD", nullable=False)
    
    # Status
    status: Mapped[InvestmentStatus] = mapped_column(
        String(50),
        default=InvestmentStatus.PENDING,
        nullable=False,
        index=True
    )
    
    # Timeline
    subscription_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    approval_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    activation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    redemption_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Units and NAV
    units_held: Mapped[Decimal] = mapped_column(Numeric(28, 18), nullable=False, default=Decimal("0"))
    entry_nav: Mapped[Optional[Decimal]] = mapped_column(MoneyType, nullable=True)
    current_nav: Mapped[Optional[Decimal]] = mapped_column(MoneyType, nullable=True)
    
    # Current value and P&L (calculated, cached)
    current_value: Mapped[Optional[Decimal]] = mapped_column(MoneyType, nullable=True)
    pnl: Mapped[Optional[Decimal]] = mapped_column(MoneyType, nullable=True)  # Absolute P&L
    pnl_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)  # % P&L
    
    last_valuation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Transaction reference
    transaction_ref: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    
    # Wire transfer / payment details
    payment_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    payment_received: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    payment_received_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Redemption details
    redemption_units_requested: Mapped[Optional[Decimal]] = mapped_column(Numeric(28, 18), nullable=True)
    redemption_requested_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    redemption_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Approval metadata
    approved_by_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=True
    )
    rejection_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Subscription agreement
    agreement_signed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    agreement_signed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    agreement_ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    
    # Accredited investor attestation
    accredited_attestation: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    accredited_attestation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    investor: Mapped["User"] = relationship("User", back_populates="investments", foreign_keys=[investor_id])
    fund: Mapped["Fund"] = relationship("Fund", back_populates="investments")
    approved_by: Mapped[Optional["User"]] = relationship("User", foreign_keys=[approved_by_id])
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_investments_investor_status', 'investor_id', 'status'),
        Index('ix_investments_fund_status', 'fund_id', 'status'),
        Index('ix_investments_subscription_date', 'subscription_date'),
    )
    
    def __repr__(self) -> str:
        return f"<Investment(id={self.id}, investor={self.investor_id}, fund={self.fund_id}, status={self.status})>"
    
    def calculate_value(self) -> Optional[Decimal]:
        """Calculate current value based on NAV and units"""
        if self.current_nav and self.units_held:
            return self.current_nav * self.units_held
        return None
    
    def calculate_pnl(self) -> Optional[Decimal]:
        """Calculate absolute P&L"""
        current = self.calculate_value()
        if current and self.amount:
            return current - self.amount
        return None


class CapitalCallStatus(str, Enum):
    """Capital call status"""
    OPEN = "open"
    PARTIAL = "partial"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"


class CapitalCall(Base, TimestampMixin):
    """
    Capital call for additional funds from investors
    Used for private equity, real estate, venture capital funds
    """
    
    __tablename__ = "capital_calls"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    fund_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("funds.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Amount details
    amount: Mapped[Decimal] = mapped_column(MoneyType, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="USD", nullable=False)
    
    # Timeline
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    # Status
    status: Mapped[CapitalCallStatus] = mapped_column(
        String(50),
        default=CapitalCallStatus.OPEN,
        nullable=False
    )
    
    amount_received: Mapped[Decimal] = mapped_column(MoneyType, default=Decimal("0"), nullable=False)
    
    # Description
    purpose: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Notification tracking
    notifications_sent: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    last_notification_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    fund: Mapped["Fund"] = relationship("Fund", back_populates="capital_calls")
    
    def __repr__(self) -> str:
        return f"<CapitalCall(fund={self.fund_id}, amount={self.amount}, due={self.due_date})>"
    
    @property
    def amount_outstanding(self) -> Decimal:
        """Calculate remaining amount to be called"""
        return self.amount - self.amount_received


class DistributionType(str, Enum):
    """Type of distribution to investors"""
    INCOME = "income"                    # Dividends, interest
    RETURN_OF_CAPITAL = "return_of_capital"
    CAPITAL_GAIN = "capital_gain"        # Profits from asset sales
    LIQUIDATION = "liquidation"          # Final distribution on fund close


class Distribution(Base, TimestampMixin):
    """
    Distribution of returns to investors
    """
    
    __tablename__ = "distributions"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    fund_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("funds.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    investor_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=True  # Null if fund-wide distribution
    )
    
    # Amount
    amount: Mapped[Decimal] = mapped_column(MoneyType, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="USD", nullable=False)
    
    # Type
    distribution_type: Mapped[DistributionType] = mapped_column(String(50), nullable=False)
    
    # Period this distribution covers
    period_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # Payment details
    paid_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    payment_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    transaction_ref: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Per-investor details
    units_at_distribution: Mapped[Optional[Decimal]] = mapped_column(Numeric(28, 18), nullable=True)
    nav_at_distribution: Mapped[Optional[Decimal]] = mapped_column(MoneyType, nullable=True)
    
    # Withholding / tax
    tax_withheld: Mapped[Optional[Decimal]] = mapped_column(MoneyType, nullable=True)
    
    # Description
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    fund: Mapped["Fund"] = relationship("Fund", back_populates="distributions")
    investor: Mapped[Optional["User"]] = relationship("User")
    
    def __repr__(self) -> str:
        return f"<Distribution(fund={self.fund_id}, type={self.distribution_type}, amount={self.amount})>"
