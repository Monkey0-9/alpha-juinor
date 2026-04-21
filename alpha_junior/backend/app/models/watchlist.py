"""
Watchlist model
Investor's saved funds for tracking
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.fund import Fund


class Watchlist(Base):
    """
    Investor's watchlist - funds they want to track
    """
    
    __tablename__ = "watchlists"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    investor_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    fund_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("funds.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    investor: Mapped["User"] = relationship("User", back_populates="watchlists")
    fund: Mapped["Fund"] = relationship("Fund", back_populates="watchlists")
    
    # Unique constraint: one watchlist entry per investor-fund pair
    __table_args__ = (
        UniqueConstraint('investor_id', 'fund_id', name='uq_watchlist_investor_fund'),
    )
    
    def __repr__(self) -> str:
        return f"<Watchlist(investor={self.investor_id}, fund={self.fund_id})>"
