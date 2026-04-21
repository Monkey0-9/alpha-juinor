"""
Fund, FundPerformance, FundDocument, FeeStructure models
Core fund management and performance tracking
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, String, Text, Numeric, Index, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, SoftDeleteMixin, MoneyType, PercentType

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.investment import Investment, CapitalCall, Distribution
    from app.models.kyc import KYCSubmission
    from app.models.watchlist import Watchlist
    from app.models.messages import Message
    from app.models.reports import Report


class FundStrategy(str, Enum):
    """Fund investment strategies"""
    LONG_SHORT = "long_short"
    MACRO = "macro"
    QUANT = "quant"
    MULTI_STRATEGY = "multi_strategy"
    PRIVATE_EQUITY = "private_equity"
    REAL_ESTATE = "real_estate"
    VENTURE_CAPITAL = "venture_capital"
    DEBT = "debt"
    COMMODITIES = "commodities"
    CRYPTOCURRENCY = "cryptocurrency"


class FundAssetClass(str, Enum):
    """Fund asset classes"""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    ALTERNATIVE = "alternative"
    MULTI_ASSET = "multi_asset"
    REAL_ASSETS = "real_assets"


class FundStatus(str, Enum):
    """Fund lifecycle status"""
    DRAFT = "draft"           # Not yet published
    ACTIVE = "active"         # Open for investment
    CLOSED = "closed"         # No longer accepting new investors
    PAUSED = "paused"         # Temporarily not accepting investments
    WINDING_DOWN = "winding_down"  # Liquidating


class FundCurrency(str, Enum):
    """Supported currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    SGD = "SGD"


class RedemptionFrequency(str, Enum):
    """How often can investors redeem"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUAL = "semi_annual"
    ANNUAL = "annual"
    LOCKED = "locked"  # No redemptions until fund closes


class Fund(Base, TimestampMixin, SoftDeleteMixin):
    """
    Core fund model
    Represents an investment fund managed on the platform
    """
    
    __tablename__ = "funds"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Manager relationship
    manager_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )
    
    # Basic info
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Strategy & classification
    strategy: Mapped[FundStrategy] = mapped_column(String(50), nullable=False, index=True)
    asset_class: Mapped[FundAssetClass] = mapped_column(String(50), nullable=False)
    
    # Currency & denomination
    currency: Mapped[FundCurrency] = mapped_column(String(10), default=FundCurrency.USD, nullable=False)
    
    # Investment terms
    min_investment: Mapped[Decimal] = mapped_column(MoneyType, nullable=False)
    target_raise: Mapped[Optional[Decimal]] = mapped_column(MoneyType, nullable=True)
    total_raised: Mapped[Decimal] = mapped_column(MoneyType, default=Decimal("0"), nullable=False)
    
    # NAV tracking
    nav: Mapped[Decimal] = mapped_column(MoneyType, nullable=False)
    nav_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Fund lifecycle
    inception_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[FundStatus] = mapped_column(
        String(50),
        default=FundStatus.DRAFT,
        nullable=False,
        index=True
    )
    
    # Fee structure
    fee_structure_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("fee_structures.id"),
        nullable=True
    )
    
    # Visibility
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Soft metrics (cached for performance)
    investor_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    aum: Mapped[Decimal] = mapped_column(MoneyType, default=Decimal("0"), nullable=False)
    
    # Relationships
    manager: Mapped["User"] = relationship("User", back_populates="managed_funds")
    
    fee_structure: Mapped[Optional["FeeStructure"]] = relationship(
        "FeeStructure",
        back_populates="funds",
        lazy="joined"
    )
    
    investments: Mapped[List["Investment"]] = relationship(
        "Investment",
        back_populates="fund",
        lazy="selectin"
    )
    
    performance_history: Mapped[List["FundPerformance"]] = relationship(
        "FundPerformance",
        back_populates="fund",
        lazy="selectin",
        order_by="desc(FundPerformance.date)"
    )
    
    documents: Mapped[List["FundDocument"]] = relationship(
        "FundDocument",
        back_populates="fund",
        lazy="selectin"
    )
    
    capital_calls: Mapped[List["CapitalCall"]] = relationship(
        "CapitalCall",
        back_populates="fund",
        lazy="selectin"
    )
    
    distributions: Mapped[List["Distribution"]] = relationship(
        "Distribution",
        back_populates="fund",
        lazy="selectin"
    )
    
    watchlists: Mapped[List["Watchlist"]] = relationship(
        "Watchlist",
        back_populates="fund",
        lazy="selectin"
    )
    
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="fund",
        lazy="selectin"
    )
    
    reports: Mapped[List["Report"]] = relationship(
        "Report",
        back_populates="fund",
        lazy="selectin"
    )
    
    # Indexes
    __table_args__ = (
        Index('ix_funds_status_strategy', 'status', 'strategy'),
        Index('ix_funds_is_public_status', 'is_public', 'status'),
        Index('ix_funds_min_investment', 'min_investment'),
    )
    
    def __repr__(self) -> str:
        return f"<Fund(id={self.id}, name={self.name}, status={self.status})>"
    
    @property
    def current_aum(self) -> Decimal:
        """Calculate current AUM"""
        return sum(inv.current_value or Decimal("0") for inv in self.investments if inv.status == "active")
    
    @property
    def is_open_for_investment(self) -> bool:
        """Check if fund is accepting new investments"""
        return self.status == FundStatus.ACTIVE and not self.is_deleted


class FundPerformance(Base):
    """
    Historical NAV and performance metrics
    Daily/weekly snapshot of fund performance
    """
    
    __tablename__ = "fund_performance"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    fund_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("funds.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Date of performance record
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    
    # NAV
    nav: Mapped[Decimal] = mapped_column(MoneyType, nullable=False)
    
    # Returns
    monthly_return: Mapped[Optional[Decimal]] = mapped_column(PercentType, nullable=True)
    ytd_return: Mapped[Optional[Decimal]] = mapped_column(PercentType, nullable=True)
    since_inception_return: Mapped[Optional[Decimal]] = mapped_column(PercentType, nullable=True)
    
    # Risk metrics
    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    max_drawdown: Mapped[Optional[Decimal]] = mapped_column(PercentType, nullable=True)
    volatility: Mapped[Optional[Decimal]] = mapped_column(PercentType, nullable=True)
    
    # Benchmark comparison
    benchmark_return: Mapped[Optional[Decimal]] = mapped_column(PercentType, nullable=True)
    alpha: Mapped[Optional[Decimal]] = mapped_column(PercentType, nullable=True)
    beta: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    
    # Additional metrics
    information_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    sortino_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4), nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    fund: Mapped["Fund"] = relationship("Fund", back_populates="performance_history")
    
    # Unique constraint: one record per fund per date
    __table_args__ = (
        Index('ix_fund_performance_fund_date', 'fund_id', 'date', unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<FundPerformance(fund_id={self.fund_id}, date={self.date}, nav={self.nav})>"


class FundDocumentType(str, Enum):
    """Types of fund documents"""
    PITCH_DECK = "pitch_deck"
    PPM = "ppm"  # Private Placement Memorandum
    SUBSCRIPTION_AGREEMENT = "subscription_agreement"
    AUDITED_FINANCIALS = "audited_financials"
    FACT_SHEET = "fact_sheet"
    DUE_DILIGENCE = "due_diligence"
    QUARTERLY_REPORT = "quarterly_report"
    ANNUAL_REPORT = "annual_report"
    LEGAL_DOCUMENT = "legal_document"
    OTHER = "other"


class FundDocument(Base, TimestampMixin):
    """
    Fund documents stored in S3
    Access controlled by investment status
    """
    
    __tablename__ = "fund_documents"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    fund_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("funds.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    doc_type: Mapped[FundDocumentType] = mapped_column(String(50), nullable=False, index=True)
    
    # S3 storage
    s3_key: Mapped[str] = mapped_column(String(500), nullable=False)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)  # bytes
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Versioning
    version: Mapped[int] = mapped_column(BigInteger, default=1, nullable=False)
    
    # Access control
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    requires_investment: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Timestamps
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    uploaded_by_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=False
    )
    
    # Soft delete (documents can be deprecated)
    is_deprecated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Relationships
    fund: Mapped["Fund"] = relationship("Fund", back_populates="documents")
    uploaded_by: Mapped["User"] = relationship("User")
    
    def __repr__(self) -> str:
        return f"<FundDocument(fund_id={self.fund_id}, type={self.doc_type}, version={self.version})>"
    
    @property
    def s3_uri(self) -> str:
        """Generate S3 URI"""
        return f"s3://{self.s3_bucket}/{self.s3_key}"


class FeeStructure(Base):
    """
    Fee structure template for funds
    Can be reused across multiple funds
    """
    
    __tablename__ = "fee_structures"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # Management fee (annual, % of AUM)
    management_fee_pct: Mapped[Decimal] = mapped_column(PercentType, nullable=False)
    
    # Performance fee (% of profits)
    performance_fee_pct: Mapped[Decimal] = mapped_column(PercentType, nullable=False)
    
    # Hurdle rate (minimum return before performance fee kicks in)
    hurdle_rate: Mapped[Optional[Decimal]] = mapped_column(PercentType, nullable=True)
    
    # High watermark (don't charge fee until previous losses recovered)
    high_watermark: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Lock-up period
    lock_up_period_months: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    
    # Redemption terms
    redemption_frequency: Mapped[RedemptionFrequency] = mapped_column(
        String(50),
        default=RedemptionFrequency.QUARTERLY,
        nullable=False
    )
    notice_period_days: Mapped[int] = mapped_column(BigInteger, default=30, nullable=False)
    
    # Metadata
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # e.g., "Standard 2/20"
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    is_template: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    funds: Mapped[List["Fund"]] = relationship("Fund", back_populates="fee_structure")
    
    def __repr__(self) -> str:
        return (
            f"<FeeStructure({self.management_fee_pct}%/{self.performance_fee_pct}%, "
            f"lockup={self.lock_up_period_months}mo)>"
        )
