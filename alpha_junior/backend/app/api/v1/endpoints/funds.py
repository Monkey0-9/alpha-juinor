"""
Fund management endpoints
Discovery, CRUD, performance metrics, documents
"""

from typing import Optional, List
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.api.deps import (
    get_db, CurrentUser, ManagerUser, AdminUser,
    require_kyc_approved, require_role
)
from app.models.fund import (
    Fund, FundPerformance, FundDocument, FeeStructure,
    FundStatus, FundStrategy, FundAssetClass
)
from app.models.user import User, UserRole
from app.models.watchlist import Watchlist
from app.models.audit import AuditLog, AuditAction

router = APIRouter(prefix="/funds", tags=["Funds"])


# Pydantic schemas
class FundListItem(BaseModel):
    id: int
    name: str
    slug: str
    strategy: str
    asset_class: str
    currency: str
    min_investment: float
    nav: float
    ytd_return: Optional[float] = None
    status: str
    is_public: bool


class FundDetail(BaseModel):
    id: int
    name: str
    slug: str
    description: Optional[str]
    strategy: str
    asset_class: str
    currency: str
    min_investment: float
    target_raise: Optional[float]
    total_raised: float
    nav: float
    inception_date: Optional[date]
    status: str
    is_public: bool
    aum: float
    investor_count: int


class FundCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=255)
    slug: str = Field(..., min_length=3, max_length=255, regex=r"^[a-z0-9-]+$")
    description: Optional[str] = None
    strategy: FundStrategy
    asset_class: FundAssetClass
    currency: str = "USD"
    min_investment: float = Field(..., gt=0)
    target_raise: Optional[float] = None
    nav: float = Field(..., gt=0)
    inception_date: Optional[date] = None
    is_public: bool = False


class PerformanceData(BaseModel):
    date: date
    nav: float
    monthly_return: Optional[float]
    ytd_return: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]


# Helper functions
async def check_fund_access(user: User, fund: Fund) -> bool:
    """Check if user can access fund details"""
    if fund.is_public and fund.status == FundStatus.ACTIVE:
        return True
    if user.is_admin or user.id == fund.manager_id:
        return True
    # Check if investor has investment in fund
    # TODO: Check investments
    return False


# Public endpoints
@router.get("", response_model=dict)
async def list_funds(
    strategy: Optional[FundStrategy] = None,
    asset_class: Optional[FundAssetClass] = None,
    min_investment_max: Optional[float] = None,
    ytd_return_min: Optional[float] = None,
    status: FundStatus = FundStatus.ACTIVE,
    is_public: bool = True,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Public fund discovery with filters and pagination
    """
    query = select(Fund).where(
        Fund.status == status,
        Fund.is_public == is_public,
        Fund.is_deleted == False
    )
    
    # Apply filters
    if strategy:
        query = query.where(Fund.strategy == strategy)
    if asset_class:
        query = query.where(Fund.asset_class == asset_class)
    if min_investment_max:
        query = query.where(Fund.min_investment <= min_investment_max)
    if search:
        search_filter = or_(
            Fund.name.ilike(f"%{search}%"),
            Fund.description.ilike(f"%{search}%")
        )
        query = query.where(search_filter)
    
    # Get total count
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar()
    
    # Apply pagination
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    funds = result.scalars().all()
    
    # Format response
    items = []
    for fund in funds:
        items.append({
            "id": fund.id,
            "name": fund.name,
            "slug": fund.slug,
            "strategy": fund.strategy.value,
            "asset_class": fund.asset_class.value,
            "currency": fund.currency,
            "min_investment": float(fund.min_investment),
            "nav": float(fund.nav),
            "status": fund.status.value,
            "is_public": fund.is_public,
        })
    
    return {
        "success": True,
        "data": {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size,
        },
        "error": None
    }


@router.get("/{slug}", response_model=dict)
async def get_fund(
    slug: str,
    current_user: Optional[User] = Depends(require_role([UserRole.INVESTOR, UserRole.MANAGER, UserRole.ADMIN])),
    db: AsyncSession = Depends(get_db)
):
    """
    Get fund details by slug
    """
    result = await db.execute(
        select(Fund).where(Fund.slug == slug, Fund.is_deleted == False)
    )
    fund = result.scalar_one_or_none()
    
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    # Check access
    if not fund.is_public and not current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not await check_fund_access(current_user, fund):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "success": True,
        "data": {
            "id": fund.id,
            "name": fund.name,
            "slug": fund.slug,
            "description": fund.description,
            "strategy": fund.strategy.value,
            "asset_class": fund.asset_class.value,
            "currency": fund.currency,
            "min_investment": float(fund.min_investment),
            "target_raise": float(fund.target_raise) if fund.target_raise else None,
            "total_raised": float(fund.total_raised),
            "nav": float(fund.nav),
            "inception_date": fund.inception_date,
            "status": fund.status.value,
            "is_public": fund.is_public,
            "aum": float(fund.aum),
            "investor_count": fund.investor_count,
            "manager": {
                "id": fund.manager.id if fund.manager else None,
                "name": fund.manager.profile.full_name if fund.manager and fund.manager.profile else None,
            }
        },
        "error": None
    }


@router.get("/{fund_id}/performance", response_model=dict)
async def get_fund_performance(
    fund_id: int,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get fund NAV history and performance metrics
    """
    result = await db.execute(
        select(Fund).where(Fund.id == fund_id, Fund.is_deleted == False)
    )
    fund = result.scalar_one_or_none()
    
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    # Query performance history
    perf_query = select(FundPerformance).where(FundPerformance.fund_id == fund_id)
    
    if from_date:
        perf_query = perf_query.where(FundPerformance.date >= from_date)
    if to_date:
        perf_query = perf_query.where(FundPerformance.date <= to_date)
    
    perf_query = perf_query.order_by(FundPerformance.date.desc())
    
    perf_result = await db.execute(perf_query)
    performances = perf_result.scalars().all()
    
    data = [{
        "date": p.date,
        "nav": float(p.nav),
        "monthly_return": float(p.monthly_return) if p.monthly_return else None,
        "ytd_return": float(p.ytd_return) if p.ytd_return else None,
        "sharpe_ratio": float(p.sharpe_ratio) if p.sharpe_ratio else None,
        "max_drawdown": float(p.max_drawdown) if p.max_drawdown else None,
    } for p in performances]
    
    return {
        "success": True,
        "data": data,
        "error": None
    }


# Manager endpoints
@router.post("", response_model=dict)
async def create_fund(
    fund_in: FundCreate,
    current_user: ManagerUser,
    db: AsyncSession = Depends(get_db)
):
    """
    Create new fund (manager only)
    """
    # Check slug uniqueness
    result = await db.execute(select(Fund).where(Fund.slug == fund_in.slug))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Slug already exists")
    
    fund = Fund(
        manager_id=current_user.id,
        name=fund_in.name,
        slug=fund_in.slug,
        description=fund_in.description,
        strategy=fund_in.strategy,
        asset_class=fund_in.asset_class,
        currency=fund_in.currency,
        min_investment=fund_in.min_investment,
        target_raise=fund_in.target_raise,
        nav=fund_in.nav,
        inception_date=fund_in.inception_date,
        status=FundStatus.DRAFT,
        is_public=fund_in.is_public,
    )
    
    db.add(fund)
    await db.flush()
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.FUND_CREATED,
        target_type="fund",
        target_id=fund.id,
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"id": fund.id, "message": "Fund created successfully"},
        "error": None
    }


@router.patch("/{fund_id}", response_model=dict)
async def update_fund(
    fund_id: int,
    fund_update: FundCreate,
    current_user: ManagerUser,
    db: AsyncSession = Depends(get_db)
):
    """
    Update fund (manager/admin only)
    """
    result = await db.execute(
        select(Fund).where(Fund.id == fund_id, Fund.is_deleted == False)
    )
    fund = result.scalar_one_or_none()
    
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    # Check ownership or admin
    if fund.manager_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized to update this fund")
    
    # Update fields
    for field, value in fund_update.dict(exclude_unset=True).items():
        setattr(fund, field, value)
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.FUND_UPDATED,
        target_type="fund",
        target_id=fund.id,
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Fund updated successfully"},
        "error": None
    }


@router.post("/{fund_id}/publish", response_model=dict)
async def publish_fund(
    fund_id: int,
    current_user: ManagerUser,
    db: AsyncSession = Depends(get_db)
):
    """
    Publish fund to make it active and discoverable
    """
    result = await db.execute(
        select(Fund).where(Fund.id == fund_id, Fund.is_deleted == False)
    )
    fund = result.scalar_one_or_none()
    
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    if fund.manager_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    fund.status = FundStatus.ACTIVE
    fund.is_public = True
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.FUND_PUBLISHED,
        target_type="fund",
        target_id=fund.id,
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Fund published successfully"},
        "error": None
    }


@router.post("/{fund_id}/nav", response_model=dict)
async def update_nav(
    fund_id: int,
    nav: float,
    current_user: ManagerUser,
    db: AsyncSession = Depends(get_db)
):
    """
    Update fund NAV (manager only)
    Also creates performance record
    """
    result = await db.execute(
        select(Fund).where(Fund.id == fund_id, Fund.is_deleted == False)
    )
    fund = result.scalar_one_or_none()
    
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    if fund.manager_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    old_nav = fund.nav
    fund.nav = nav
    fund.nav_updated_at = datetime.utcnow()
    
    # Create performance record
    perf = FundPerformance(
        fund_id=fund_id,
        date=date.today(),
        nav=nav,
        # Calculate returns if previous data exists
    )
    db.add(perf)
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.FUND_NAV_UPDATED,
        target_type="fund",
        target_id=fund.id,
        payload={"before": {"nav": float(old_nav)}, "after": {"nav": float(nav)}}
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "NAV updated successfully", "new_nav": nav},
        "error": None
    }


# Watchlist endpoints
@router.post("/{fund_id}/watch", response_model=dict)
async def add_to_watchlist(
    fund_id: int,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """Add fund to investor watchlist"""
    # Check fund exists
    result = await db.execute(
        select(Fund).where(Fund.id == fund_id, Fund.is_deleted == False)
    )
    fund = result.scalar_one_or_none()
    
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    # Check not already in watchlist
    result = await db.execute(
        select(Watchlist).where(
            Watchlist.investor_id == current_user.id,
            Watchlist.fund_id == fund_id
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Already in watchlist")
    
    watchlist_item = Watchlist(investor_id=current_user.id, fund_id=fund_id)
    db.add(watchlist_item)
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Added to watchlist"},
        "error": None
    }


@router.delete("/{fund_id}/watch", response_model=dict)
async def remove_from_watchlist(
    fund_id: int,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """Remove fund from investor watchlist"""
    result = await db.execute(
        select(Watchlist).where(
            Watchlist.investor_id == current_user.id,
            Watchlist.fund_id == fund_id
        )
    )
    watchlist_item = result.scalar_one_or_none()
    
    if watchlist_item:
        await db.delete(watchlist_item)
        await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Removed from watchlist"},
        "error": None
    }
