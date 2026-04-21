"""
Investment endpoints
Subscription, redemption, portfolio management
"""

from typing import Optional
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from decimal import Decimal

from app.api.deps import (
    get_db, CurrentUser, CurrentUserWithKYC, ManagerUser,
    require_kyc_approved
)
from app.models.investment import Investment, InvestmentStatus
from app.models.fund import Fund, FundStatus
from app.models.audit import AuditLog, AuditAction

router = APIRouter()


class InvestmentCreate(BaseModel):
    fund_id: int
    amount: float = Field(..., gt=0)
    currency: str = "USD"


class InvestmentResponse(BaseModel):
    id: int
    fund_id: int
    fund_name: str
    amount: float
    status: str
    subscription_date: date
    current_value: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]


@router.get("/portfolio", response_model=dict)
async def get_portfolio(
    current_user: CurrentUser = Depends(require_kyc_approved),
    db: AsyncSession = Depends(get_db)
):
    """
    Get investor's complete portfolio
    Requires KYC approval
    """
    result = await db.execute(
        select(Investment).where(
            Investment.investor_id == current_user.id,
            Investment.is_deleted == False
        ).order_by(Investment.subscription_date.desc())
    )
    investments = result.scalars().all()
    
    items = []
    total_invested = Decimal("0")
    total_value = Decimal("0")
    
    for inv in investments:
        items.append({
            "id": inv.id,
            "fund_id": inv.fund_id,
            "fund_name": inv.fund.name if inv.fund else "Unknown",
            "amount": float(inv.amount),
            "status": inv.status.value,
            "subscription_date": inv.subscription_date.date(),
            "current_value": float(inv.current_value) if inv.current_value else None,
            "pnl": float(inv.pnl) if inv.pnl else None,
            "pnl_pct": float(inv.pnl_pct) if inv.pnl_pct else None,
        })
        
        total_invested += inv.amount
        if inv.current_value:
            total_value += inv.current_value
    
    total_pnl = total_value - total_invested if total_value > 0 else Decimal("0")
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    
    return {
        "success": True,
        "data": {
            "summary": {
                "total_invested": float(total_invested),
                "current_value": float(total_value) if total_value > 0 else None,
                "total_pnl": float(total_pnl),
                "total_pnl_pct": float(total_pnl_pct),
                "active_investments": len([i for i in investments if i.status == InvestmentStatus.ACTIVE]),
            },
            "investments": items,
        },
        "error": None
    }


@router.post("/subscribe", response_model=dict)
async def subscribe(
    investment_in: InvestmentCreate,
    current_user: CurrentUser = Depends(require_kyc_approved),
    db: AsyncSession = Depends(get_db)
):
    """
    Subscribe to a fund
    Creates pending investment awaiting manager approval
    """
    # Validate fund
    result = await db.execute(
        select(Fund).where(
            Fund.id == investment_in.fund_id,
            Fund.is_deleted == False
        )
    )
    fund = result.scalar_one_or_none()
    
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    if fund.status != FundStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Fund is not active")
    
    if investment_in.amount < float(fund.min_investment):
        raise HTTPException(
            status_code=400,
            detail=f"Minimum investment is {fund.min_investment} {fund.currency}"
        )
    
    # Check for existing pending investment
    result = await db.execute(
        select(Investment).where(
            Investment.investor_id == current_user.id,
            Investment.fund_id == investment_in.fund_id,
            Investment.status.in_([InvestmentStatus.PENDING, InvestmentStatus.APPROVED, InvestmentStatus.ACTIVE])
        )
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail="You already have an active or pending investment in this fund"
        )
    
    # Create investment
    investment = Investment(
        investor_id=current_user.id,
        fund_id=investment_in.fund_id,
        amount=Decimal(str(investment_in.amount)),
        currency=investment_in.currency,
        status=InvestmentStatus.PENDING,
        units_held=Decimal("0"),  # Will be calculated on activation
        accredited_attestation=True,  # User confirms accreditation during submission
    )
    
    db.add(investment)
    await db.flush()
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.INVESTMENT_SUBMITTED,
        target_type="investment",
        target_id=investment.id,
        payload={"fund_id": fund.id, "amount": investment_in.amount}
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {
            "investment_id": investment.id,
            "status": "pending",
            "message": "Investment submitted for approval",
        },
        "error": None
    }


@router.post("/{investment_id}/approve", response_model=dict)
async def approve_investment(
    investment_id: int,
    current_user: ManagerUser,
    db: AsyncSession = Depends(get_db)
):
    """
    Manager approves pending investment
    """
    result = await db.execute(
        select(Investment).where(
            Investment.id == investment_id,
            Investment.is_deleted == False
        )
    )
    investment = result.scalar_one_or_none()
    
    if not investment:
        raise HTTPException(status_code=404, detail="Investment not found")
    
    # Check manager owns the fund
    if investment.fund.manager_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if investment.status != InvestmentStatus.PENDING:
        raise HTTPException(status_code=400, detail="Investment is not pending")
    
    # Approve
    investment.status = InvestmentStatus.APPROVED
    investment.approval_date = date.today()
    investment.approved_by_id = current_user.id
    
    # Calculate units
    if investment.fund.nav and investment.fund.nav > 0:
        investment.units_held = investment.amount / investment.fund.nav
        investment.entry_nav = investment.fund.nav
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.INVESTMENT_APPROVED,
        target_type="investment",
        target_id=investment.id,
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Investment approved"},
        "error": None
    }


@router.post("/{investment_id}/reject", response_model=dict)
async def reject_investment(
    investment_id: int,
    reason: str,
    current_user: ManagerUser,
    db: AsyncSession = Depends(get_db)
):
    """Manager rejects pending investment"""
    result = await db.execute(
        select(Investment).where(Investment.id == investment_id)
    )
    investment = result.scalar_one_or_none()
    
    if not investment:
        raise HTTPException(status_code=404, detail="Investment not found")
    
    if investment.fund.manager_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    investment.status = InvestmentStatus.REJECTED
    investment.rejection_reason = reason
    
    await db.commit()
    
    return {"success": True, "data": {"message": "Investment rejected"}, "error": None}


@router.post("/{investment_id}/redeem", response_model=dict)
async def request_redemption(
    investment_id: int,
    units: Optional[float] = None,  # None = full redemption
    reason: Optional[str] = None,
    current_user: CurrentUser = Depends(require_kyc_approved),
    db: AsyncSession = Depends(get_db)
):
    """
    Investor requests redemption
    """
    result = await db.execute(
        select(Investment).where(
            Investment.id == investment_id,
            Investment.investor_id == current_user.id,
            Investment.is_deleted == False
        )
    )
    investment = result.scalar_one_or_none()
    
    if not investment:
        raise HTTPException(status_code=404, detail="Investment not found")
    
    if investment.status != InvestmentStatus.ACTIVE:
        raise HTTPException(status_code=400, detail="Investment is not active")
    
    # Check lock-up period
    # TODO: Implement lock-up check based on fund fee_structure
    
    # Set redemption request
    investment.status = InvestmentStatus.REDEEMING
    investment.redemption_units_requested = Decimal(str(units)) if units else investment.units_held
    investment.redemption_requested_at = date.today()
    investment.redemption_reason = reason
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.REDEMPTION_REQUESTED,
        target_type="investment",
        target_id=investment.id,
        payload={"units_requested": str(units) if units else "full"}
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Redemption request submitted"},
        "error": None
    }


@router.get("/history", response_model=dict)
async def get_investment_history(
    current_user: CurrentUser,
    status: Optional[InvestmentStatus] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Get investment history with pagination"""
    query = select(Investment).where(
        Investment.investor_id == current_user.id,
        Investment.is_deleted == False
    )
    
    if status:
        query = query.where(Investment.status == status)
    
    query = query.order_by(Investment.subscription_date.desc())
    
    # Count total
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar()
    
    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    investments = result.scalars().all()
    
    items = [{
        "id": inv.id,
        "fund_id": inv.fund_id,
        "fund_name": inv.fund.name if inv.fund else "Unknown",
        "amount": float(inv.amount),
        "status": inv.status.value,
        "subscription_date": inv.subscription_date.date(),
        "current_value": float(inv.current_value) if inv.current_value else None,
    } for inv in investments]
    
    return {
        "success": True,
        "data": {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        },
        "error": None
    }
