"""
User management endpoints
Profile, settings, notifications
"""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr

from app.api.deps import get_db, CurrentUser
from app.models.user import User, UserProfile, AccreditationStatus
from app.models.notifications import Notification, NotificationType
from app.models.audit import AuditLog, AuditAction

router = APIRouter()


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    country: Optional[str] = None
    bio: Optional[str] = None
    linkedin_url: Optional[str] = None
    company_name: Optional[str] = None
    job_title: Optional[str] = None


class AccreditationUpdate(BaseModel):
    accreditation_status: AccreditationStatus
    net_worth_bracket: Optional[str] = None
    annual_income_bracket: Optional[str] = None


@router.get("/profile", response_model=dict)
async def get_profile(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """Get current user's profile"""
    profile = current_user.profile
    
    return {
        "success": True,
        "data": {
            "user": {
                "id": current_user.id,
                "email": current_user.email,
                "role": current_user.role.value,
                "kyc_status": current_user.kyc_status.value,
                "is_2fa_enabled": current_user.is_2fa_enabled,
                "created_at": current_user.created_at,
                "last_login": current_user.last_login,
            },
            "profile": {
                "full_name": profile.full_name if profile else None,
                "country": profile.country if profile else None,
                "accreditation_status": profile.accreditation_status.value if profile else None,
                "net_worth_bracket": profile.net_worth_bracket if profile else None,
                "avatar_url": profile.avatar_url if profile else None,
                "bio": profile.bio if profile else None,
                "linkedin_url": profile.linkedin_url if profile else None,
                "company_name": profile.company_name if profile else None,
                "job_title": profile.job_title if profile else None,
            } if profile else None,
        },
        "error": None
    }


@router.patch("/profile", response_model=dict)
async def update_profile(
    profile_update: ProfileUpdate,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """Update user profile"""
    # Create profile if doesn't exist
    if not current_user.profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
        await db.flush()
    else:
        profile = current_user.profile
    
    # Update fields
    for field, value in profile_update.dict(exclude_unset=True).items():
        setattr(profile, field, value)
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.PROFILE_UPDATED,
        target_type="user",
        target_id=current_user.id,
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Profile updated successfully"},
        "error": None
    }


@router.post("/accreditation", response_model=dict)
async def update_accreditation(
    accred: AccreditationUpdate,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """Update accreditation status (with attestation)"""
    if not current_user.profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
        await db.flush()
    else:
        profile = current_user.profile
    
    profile.accreditation_status = accred.accreditation_status
    profile.net_worth_bracket = accred.net_worth_bracket
    profile.annual_income_bracket = accred.annual_income_bracket
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.ACCREDITATION_ATTESTED,
        target_type="user",
        target_id=current_user.id,
        payload={
            "accreditation_status": accred.accreditation_status.value,
            "net_worth_bracket": accred.net_worth_bracket,
        }
    )
    db.add(audit)
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Accreditation information updated"},
        "error": None
    }


@router.post("/avatar", response_model=dict)
async def upload_avatar(
    avatar: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_db)
):
    """Upload profile avatar"""
    # TODO: Upload to S3
    # avatar_url = await upload_file_to_s3(avatar, f"avatars/{current_user.id}")
    
    # Update profile
    if not current_user.profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
    else:
        profile = current_user.profile
    
    profile.avatar_url = f"https://cdn.example.com/avatars/{current_user.id}.jpg"  # Placeholder
    await db.commit()
    
    return {
        "success": True,
        "data": {"avatar_url": profile.avatar_url},
        "error": None
    }


# Notifications endpoints
@router.get("/notifications", response_model=dict)
async def get_notifications(
    current_user: CurrentUser,
    unread_only: bool = False,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Get user notifications"""
    query = select(Notification).where(
        Notification.user_id == current_user.id
    )
    
    if unread_only:
        query = query.where(Notification.is_read == False)
    
    query = query.order_by(Notification.created_at.desc())
    
    # Count
    count_result = await db.execute(
        select(func.count()).where(Notification.user_id == current_user.id)
    )
    total = count_result.scalar()
    
    unread_count_result = await db.execute(
        select(func.count()).where(
            Notification.user_id == current_user.id,
            Notification.is_read == False
        )
    )
    unread_count = unread_count_result.scalar()
    
    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    notifications = result.scalars().all()
    
    items = [{
        "id": n.id,
        "type": n.type.value,
        "title": n.title,
        "body": n.body,
        "is_read": n.is_read,
        "created_at": n.created_at,
        "action_url": n.action_url,
    } for n in notifications]
    
    return {
        "success": True,
        "data": {
            "items": items,
            "total": total,
            "unread_count": unread_count,
            "page": page,
            "page_size": page_size,
        },
        "error": None
    }


@router.patch("/notifications/{notification_id}/read", response_model=dict)
async def mark_notification_read(
    notification_id: int,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """Mark notification as read"""
    result = await db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == current_user.id
        )
    )
    notification = result.scalar_one_or_none()
    
    if notification:
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        await db.commit()
    
    return {
        "success": True,
        "data": {"message": "Notification marked as read"},
        "error": None
    }


@router.patch("/notifications/read-all", response_model=dict)
async def mark_all_notifications_read(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """Mark all notifications as read"""
    result = await db.execute(
        select(Notification).where(
            Notification.user_id == current_user.id,
            Notification.is_read == False
        )
    )
    notifications = result.scalars().all()
    
    for n in notifications:
        n.is_read = True
        n.read_at = datetime.utcnow()
    
    await db.commit()
    
    return {
        "success": True,
        "data": {"message": "All notifications marked as read"},
        "error": None
    }
