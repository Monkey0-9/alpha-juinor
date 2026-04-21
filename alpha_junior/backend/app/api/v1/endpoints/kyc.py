"""
KYC (Know Your Customer) endpoints
Identity verification flow
"""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.api.deps import get_db, CurrentUser, AdminUser, require_kyc_approved
from app.models.kyc import KYCSubmission, KYCStatus, IDType
from app.models.user import User, KYCStatus as UserKYCStatus
from app.models.audit import AuditLog, AuditAction

router = APIRouter()


class KYCStatusResponse(BaseModel):
    status: str
    submitted_at: Optional[datetime]
    reviewed_at: Optional[datetime]
    rejection_reason: Optional[str]


class KYCReviewRequest(BaseModel):
    status: str  # approved, rejected, additional_info_required
    rejection_reason: Optional[str] = None
    additional_info_requested: Optional[str] = None


@router.get("/status", response_model=dict)
async def get_kyc_status(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """Get current user's KYC status"""
    # Get most recent KYC submission
    result = await db.execute(
        select(KYCSubmission).where(
            KYCSubmission.user_id == current_user.id
        ).order_by(KYCSubmission.submitted_at.desc())
    )
    submission = result.scalar_one_or_none()
    
    return {
        "success": True,
        "data": {
            "kyc_status": current_user.kyc_status.value,
            "submission": {
                "status": submission.status.value if submission else None,
                "submitted_at": submission.submitted_at if submission else None,
                "reviewed_at": submission.reviewed_at if submission else None,
                "rejection_reason": submission.rejection_reason if submission else None,
            } if submission else None,
        },
        "error": None
    }


@router.post("/submit", response_model=dict)
async def submit_kyc(
    id_type: str = Form(...),
    id_front: UploadFile = File(...),
    id_back: UploadFile = File(None),
    selfie: UploadFile = File(...),
    proof_of_address: UploadFile = File(None),
    id_number: Optional[str] = Form(None),
    id_country: Optional[str] = Form(None),
    current_user: CurrentUser = Depends(require_kyc_approved),
    db: AsyncSession = Depends(get_db)
):
    """
    Submit KYC documents for verification
    Uploads to S3 and creates KYC submission record
    """
    # Validate ID type
    try:
        id_type_enum = IDType(id_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID type")
    
    # TODO: Upload files to S3
    # s3_keys = await upload_kyc_documents(id_front, id_back, selfie, proof_of_address)
    
    # Create KYC submission
    submission = KYCSubmission(
        user_id=current_user.id,
        id_type=id_type_enum,
        id_number=id_number,  # Should be encrypted
        id_country=id_country,
        # S3 keys would be set here
        id_front_s3_key=f"kyc/{current_user.id}/id_front_{datetime.utcnow().timestamp()}",
        id_front_s3_bucket="alpha-junior-kyc",
        id_back_s3_key=None,  # Set if provided
        selfie_s3_key=f"kyc/{current_user.id}/selfie_{datetime.utcnow().timestamp()}",
        selfie_s3_bucket="alpha-junior-kyc",
        proof_of_address_s3_key=None,  # Set if provided
        status=KYCStatus.PENDING,
        submitted_from_ip=None,  # Set from request
    )
    
    db.add(submission)
    
    # Update user KYC status
    current_user.kyc_status = UserKYCStatus.PENDING
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.KYC_SUBMITTED,
        target_type="user",
        target_id=current_user.id,
    )
    db.add(audit)
    
    await db.commit()
    
    # TODO: Trigger async verification (AI + manual review)
    
    return {
        "success": True,
        "data": {
            "message": "KYC submitted successfully",
            "submission_id": submission.id,
            "status": "pending",
        },
        "error": None
    }


@router.get("/admin/pending", response_model=dict)
async def get_pending_kyc(
    current_user: AdminUser,
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """Admin: Get pending KYC submissions for review"""
    from sqlalchemy import func
    
    query = select(KYCSubmission).where(
        KYCSubmission.status.in_([KYCStatus.PENDING, KYCStatus.UNDER_REVIEW])
    ).order_by(KYCSubmission.submitted_at.asc())
    
    # Count
    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar()
    
    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    submissions = result.scalars().all()
    
    items = [{
        "id": sub.id,
        "user_id": sub.user_id,
        "user_email": sub.user.email,
        "user_name": sub.user.profile.full_name if sub.user.profile else None,
        "id_type": sub.id_type.value,
        "status": sub.status.value,
        "submitted_at": sub.submitted_at,
        "id_country": sub.id_country,
    } for sub in submissions]
    
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


@router.patch("/admin/{submission_id}", response_model=dict)
async def review_kyc(
    submission_id: int,
    review: KYCReviewRequest,
    current_user: AdminUser,
    db: AsyncSession = Depends(get_db)
):
    """Admin: Approve or reject KYC submission"""
    result = await db.execute(
        select(KYCSubmission).where(KYCSubmission.id == submission_id)
    )
    submission = result.scalar_one_or_none()
    
    if not submission:
        raise HTTPException(status_code=404, detail="KYC submission not found")
    
    # Update submission
    try:
        new_status = KYCStatus(review.status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid status")
    
    submission.status = new_status
    submission.rejection_reason = review.rejection_reason
    submission.additional_info_requested = review.additional_info_requested
    submission.reviewed_at = datetime.utcnow()
    submission.reviewer_id = current_user.id
    
    # Update user KYC status
    if new_status == KYCStatus.APPROVED:
        submission.user.kyc_status = UserKYCStatus.APPROVED
        audit_action = AuditAction.KYC_APPROVED
    elif new_status == KYCStatus.REJECTED:
        submission.user.kyc_status = UserKYCStatus.REJECTED
        audit_action = AuditAction.KYC_REJECTED
    else:
        audit_action = AuditAction.KYC_SUBMITTED  # Fallback
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=audit_action,
        target_type="user",
        target_id=submission.user_id,
        payload={"submission_id": submission_id, "status": review.status}
    )
    db.add(audit)
    
    await db.commit()
    
    # TODO: Send notification to user
    
    return {
        "success": True,
        "data": {
            "message": f"KYC {review.status} successfully",
            "user_id": submission.user_id,
            "new_status": review.status,
        },
        "error": None
    }
