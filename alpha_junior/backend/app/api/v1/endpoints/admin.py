"""
Admin endpoints
Platform management, user management, analytics
"""

from typing import Optional
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.api.deps import get_db, AdminUser, SuperAdminUser
from app.models.user import User, UserRole, KYCStatus
from app.models.fund import Fund, FundStatus
from app.models.investment import Investment
from app.models.audit import AuditLog, AuditAction

router = APIRouter()


class UserRoleUpdate(BaseModel):
    role: UserRole


class UserStatusUpdate(BaseModel):
    is_active: bool


# User management
@router.get("/users", response_model=dict)
async def list_users(
    current_user: AdminUser,
    role: Optional[UserRole] = None,
    kyc_status: Optional[KYCStatus] = None,
    is_active: Optional[bool] = None,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Admin: List all users with filters"""
    query = select(User).where(User.is_deleted == False)
    
    if role:
        query = query.where(User.role == role)
    if kyc_status:
        query = query.where(User.kyc_status == kyc_status)
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    if search:
        search_filter = or_(
            User.email.ilike(f"%{search}%"),
            User.profile.has(full_name=search) if hasattr(User, 'profile') else False
        )
        query = query.where(search_filter)
    
    # Count
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar()
    
    # Paginate
    query = query.order_by(User.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    items = [{
        "id": u.id,
        "email": u.email,
        "role": u.role.value,
        "kyc_status": u.kyc_status.value,
        "is_active": u.is_active,
        "created_at": u.created_at,
        "last_login": u.last_login,
        "full_name": u.profile.full_name if u.profile else None,
    } for u in users]
    
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


@router.patch("/users/{user_id}/role", response_model=dict)
async def update_user_role(
    user_id: int,
    role_update: UserRoleUpdate,
    current_user: SuperAdminUser,
    db: AsyncSession = Depends(get_db)
):
    """SuperAdmin: Change user role"""
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted == False)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    old_role = user.role
    user.role = role_update.role
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.USER_ROLE_CHANGED,
        target_type="user",
        target_id=user.id,
        payload={"before": {"role": old_role.value}, "after": {"role": role_update.role.value}}
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": f"User role updated to {role_update.role.value}"},
        "error": None
    }


@router.patch("/users/{user_id}/status", response_model=dict)
async def update_user_status(
    user_id: int,
    status_update: UserStatusUpdate,
    current_user: AdminUser,
    db: AsyncSession = Depends(get_db)
):
    """Admin: Activate/deactivate user"""
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_deleted == False)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Cannot deactivate superadmin
    if user.role == UserRole.SUPERADMIN and not current_user.is_superadmin:
        raise HTTPException(status_code=403, detail="Cannot modify superadmin")
    
    user.is_active = status_update.is_active
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.USER_DEACTIVATED if not status_update.is_active else AuditAction.USER_UPDATED,
        target_type="user",
        target_id=user.id,
        payload={"is_active": status_update.is_active}
    )
    db.add(audit)
    
    await db.commit()
    
    status_text = "activated" if status_update.is_active else "deactivated"
    return {
        "success": True,
        "data": {"message": f"User {status_text}"},
        "error": None
    }


# Platform analytics
@router.get("/analytics/platform", response_model=dict)
async def get_platform_analytics(
    current_user: AdminUser,
    db: AsyncSession = Depends(get_db)
):
    """Admin: Get platform-wide analytics"""
    # User counts
    user_counts = await db.execute(
        select(User.role, func.count(User.id)).where(User.is_deleted == False).group_by(User.role)
    )
    users_by_role = {role.value: count for role, count in user_counts.all()}
    
    total_users = sum(users_by_role.values())
    
    # KYC breakdown
    kyc_counts = await db.execute(
        select(User.kyc_status, func.count(User.id)).where(User.is_deleted == False).group_by(User.kyc_status)
    )
    kyc_breakdown = {status.value: count for status, count in kyc_counts.all()}
    
    # Fund stats
    fund_counts = await db.execute(
        select(Fund.status, func.count(Fund.id)).where(Fund.is_deleted == False).group_by(Fund.status)
    )
    funds_by_status = {status.value: count for status, count in fund_counts.all()}
    
    # Total AUM
    aum_result = await db.execute(
        select(func.sum(Fund.aum)).where(Fund.is_deleted == False, Fund.status == FundStatus.ACTIVE)
    )
    total_aum = aum_result.scalar() or 0
    
    # Investment stats
    investment_counts = await db.execute(
        select(Investment.status, func.count(Investment.id))
        .where(Investment.is_deleted == False)
        .group_by(Investment.status)
    )
    investments_by_status = {status.value: count for status, count in investment_counts.all()}
    
    # Recent signups (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_signups = await db.execute(
        select(func.count(User.id)).where(
            User.created_at >= thirty_days_ago,
            User.is_deleted == False
        )
    )
    new_users_30d = recent_signups.scalar()
    
    return {
        "success": True,
        "data": {
            "users": {
                "total": total_users,
                "by_role": users_by_role,
                "kyc_breakdown": kyc_breakdown,
                "new_last_30d": new_users_30d,
            },
            "funds": {
                "by_status": funds_by_status,
                "total_aum": float(total_aum),
            },
            "investments": {
                "by_status": investments_by_status,
            },
        },
        "error": None
    }


# Audit logs
@router.get("/audit-logs", response_model=dict)
async def get_audit_logs(
    current_user: AdminUser,
    actor_id: Optional[int] = None,
    action: Optional[str] = None,
    target_type: Optional[str] = None,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Admin: Searchable audit log"""
    query = select(AuditLog)
    
    if actor_id:
        query = query.where(AuditLog.actor_id == actor_id)
    if action:
        query = query.where(AuditLog.action == action)
    if target_type:
        query = query.where(AuditLog.target_type == target_type)
    if from_date:
        query = query.where(AuditLog.timestamp >= datetime.combine(from_date, datetime.min.time()))
    if to_date:
        query = query.where(AuditLog.timestamp <= datetime.combine(to_date, datetime.max.time()))
    
    # Count
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar()
    
    # Paginate
    query = query.order_by(AuditLog.timestamp.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    logs = result.scalars().all()
    
    items = [{
        "id": log.id,
        "actor_id": log.actor_id,
        "actor_email": log.actor.email if log.actor else None,
        "action": log.action.value if hasattr(log.action, 'value') else log.action,
        "target_type": log.target_type,
        "target_id": log.target_id,
        "ip_address": log.ip_address,
        "timestamp": log.timestamp,
        "payload": log.payload,
    } for log in logs]
    
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


# Fund management
@router.get("/funds/all", response_model=dict)
async def list_all_funds(
    current_user: AdminUser,
    status: Optional[FundStatus] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Admin: List all funds including non-public"""
    query = select(Fund).where(Fund.is_deleted == False)
    
    if status:
        query = query.where(Fund.status == status)
    
    # Count
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar()
    
    # Paginate
    query = query.order_by(Fund.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    funds = result.scalars().all()
    
    items = [{
        "id": f.id,
        "name": f.name,
        "slug": f.slug,
        "manager_id": f.manager_id,
        "manager_email": f.manager.email if f.manager else None,
        "status": f.status.value,
        "is_public": f.is_public,
        "aum": float(f.aum),
        "investor_count": f.investor_count,
        "created_at": f.created_at,
    } for f in funds]
    
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


@router.patch("/funds/{fund_id}/status", response_model=dict)
async def update_fund_status(
    fund_id: int,
    status: FundStatus,
    current_user: AdminUser,
    db: AsyncSession = Depends(get_db)
):
    """Admin: Change fund status (pause, close, etc.)"""
    result = await db.execute(
        select(Fund).where(Fund.id == fund_id, Fund.is_deleted == False)
    )
    fund = result.scalar_one_or_none()
    
    if not fund:
        raise HTTPException(status_code=404, detail="Fund not found")
    
    old_status = fund.status
    fund.status = status
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.FUND_STATUS_CHANGED,
        target_type="fund",
        target_id=fund.id,
        payload={"before": old_status.value, "after": status.value}
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": f"Fund status updated to {status.value}"},
        "error": None
    }
