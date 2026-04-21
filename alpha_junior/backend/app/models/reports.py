"""
Report model
Fund reports (monthly, quarterly, annual)
Stored in S3, delivered to investors
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional, TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.fund import Fund


class ReportPeriod(str, Enum):
    """Report period type"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    AD_HOC = "ad_hoc"


class Report(Base, TimestampMixin):
    """
    Fund report stored in S3
    PDF documents generated for investors
    """
    
    __tablename__ = "reports"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    fund_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("funds.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Report period
    period: Mapped[ReportPeriod] = mapped_column(String(50), nullable=False, index=True)
    period_date: Mapped[date] = mapped_column(Date, nullable=False)  # e.g., 2024-03-31 for Q1 2024
    
    # S3 storage
    s3_key: Mapped[str] = mapped_column(String(500), nullable=False)
    s3_bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # File metadata
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)  # bytes
    
    # Generation tracking
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    generated_by_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("users.id"),
        nullable=True
    )
    
    # Distribution tracking
    sent_to_investors_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    investors_notified_count: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    emails_delivered: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    emails_failed: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    
    # Access control
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    requires_investment: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Report metadata
    report_title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Performance highlights (cached for quick display)
    period_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
    ytd_return: Mapped[Optional[float]] = mapped_column(Numeric(10, 4), nullable=True)
    period_nav: Mapped[Optional[float]] = mapped_column(Numeric(18, 8), nullable=True)
    
    # Version control
    version: Mapped[int] = mapped_column(BigInteger, default=1, nullable=False)
    supersedes_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("reports.id"),
        nullable=True
    )
    
    # Status
    is_draft: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_final: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Relationships
    fund: Mapped["Fund"] = relationship("Fund", back_populates="reports")
    generated_by: Mapped[Optional["User"]] = relationship("User")
    
    def __repr__(self) -> str:
        return f"<Report(id={self.id}, fund={self.fund_id}, period={self.period}, date={self.period_date})>"
    
    @property
    def s3_uri(self) -> str:
        """Generate S3 URI"""
        return f"s3://{self.s3_bucket}/{self.s3_key}"
    
    @property
    def is_distributed(self) -> bool:
        """Check if report has been sent to investors"""
        return self.sent_to_investors_at is not None
    
    @property
    def display_name(self) -> str:
        """Generate human-readable report name"""
        period_str = self.period_date.strftime("%B %Y") if self.period == ReportPeriod.MONTHLY else \
                     f"Q{(self.period_date.month - 1) // 3 + 1} {self.period_date.year}" if self.period == ReportPeriod.QUARTERLY else \
                     str(self.period_date.year)
        return f"{self.fund.name} — {self.period.value.title()} Report — {period_str}"
