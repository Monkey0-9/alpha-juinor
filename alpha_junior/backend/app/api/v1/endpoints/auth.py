"""
Authentication endpoints
Registration, login, 2FA, password reset, token refresh
"""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr

from app.api.deps import get_db, get_current_user, CurrentUser, rate_limit_check
from app.core.security import (
    verify_password, get_password_hash, create_access_token, create_refresh_token,
    generate_totp_secret, get_totp_provisioning_uri, verify_totp,
    generate_email_verification_token, generate_password_reset_token,
    TokenBlacklist, RateLimiter, get_token_jti
)
from app.core.config import settings
from app.models.user import User, UserProfile, UserRole, KYCStatus
from app.models.audit import AuditLog, AuditAction

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


# Pydantic schemas
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: str | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str
    totp_code: str | None = None


class TokenResponse(BaseModel):
    success: bool = True
    data: dict
    error: None = None


class Enable2FARequest(BaseModel):
    totp_code: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str


@router.post("/register", response_model=TokenResponse)
async def register(
    request: Request,
    user_in: UserRegister,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Register new user account
    Sends verification email
    """
    # Check if email exists
    result = await db.execute(
        select(User).where(
            or_(User.email == user_in.email, User.email == user_in.email.lower())
        )
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create user
    verification_token = generate_email_verification_token()
    
    user = User(
        email=user_in.email.lower(),
        password_hash=get_password_hash(user_in.password),
        role=UserRole.INVESTOR,
        kyc_status=KYCStatus.PENDING,
        email_verification_token=verification_token,
    )
    
    db.add(user)
    await db.flush()  # Get user.id
    
    # Create profile
    profile = UserProfile(
        user_id=user.id,
        full_name=user_in.full_name,
        phone=user_in.phone,
    )
    db.add(profile)
    
    # Audit log
    audit = AuditLog(
        actor_id=user.id,
        action=AuditAction.USER_REGISTERED,
        target_type="user",
        target_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(audit)
    
    await db.commit()
    
    # Send verification email (background task)
    # background_tasks.add_task(send_verification_email, user.email, verification_token)
    
    return TokenResponse(data={
        "message": "Registration successful. Please verify your email.",
        "user_id": user.id,
    })


@router.post("/verify-email", response_model=TokenResponse)
async def verify_email(
    request: Request,
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """Verify email address with token from email"""
    result = await db.execute(
        select(User).where(User.email_verification_token == token)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token",
        )
    
    user.email_verified_at = datetime.utcnow()
    user.email_verification_token = None
    user.is_active = True
    
    # Audit log
    audit = AuditLog(
        actor_id=user.id,
        action=AuditAction.EMAIL_VERIFIED,
        target_type="user",
        target_id=user.id,
        ip_address=request.client.host if request.client else None,
    )
    db.add(audit)
    
    await db.commit()
    
    return TokenResponse(data={"message": "Email verified successfully"})


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and issue tokens
    Rate limited: 5 attempts per 15 minutes
    """
    # Rate limiting via IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Check login lockout
    redis = request.app.state.redis
    limiter = RateLimiter(redis)
    
    if await limiter.is_login_locked(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again in 15 minutes.",
        )
    
    # Find user
    result = await db.execute(
        select(User).where(User.email == credentials.email.lower())
    )
    user = result.scalar_one_or_none()
    
    # Verify credentials
    if not user or not verify_password(credentials.password, user.password_hash):
        # Record failed attempt
        await limiter.record_login_attempt(client_ip)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check 2FA if enabled
    if user.is_2fa_enabled:
        if not credentials.totp_code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="2FA code required",
            )
        
        if not verify_totp(credentials.totp_code, user.totp_secret):
            await limiter.record_login_attempt(client_ip)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA code",
            )
    
    # Check if email verified
    if not user.email_verified_at:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please check your inbox.",
        )
    
    # Clear login attempts on success
    await limiter.clear_login_attempts(client_ip)
    
    # Update last login
    user.last_login = datetime.utcnow()
    
    # Generate tokens
    access_token = create_access_token(
        subject=str(user.id),
        additional_claims={
            "email": user.email,
            "role": user.role.value,
            "kyc_status": user.kyc_status.value,
        }
    )
    
    refresh_token, refresh_expires = create_refresh_token(subject=str(user.id))
    
    # Audit log
    audit = AuditLog(
        actor_id=user.id,
        action=AuditAction.USER_LOGIN,
        target_type="user",
        target_id=user.id,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
    )
    db.add(audit)
    
    await db.commit()
    
    return TokenResponse(data={
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "refresh_expires_in": settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        "user": {
            "id": user.id,
            "email": user.email,
            "role": user.role.value,
            "kyc_status": user.kyc_status.value,
            "is_2fa_enabled": user.is_2fa_enabled,
        }
    })


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Rotate refresh token (sliding window)
    Old token is blacklisted
    """
    from app.core.security import decode_token
    
    payload = decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    user_id = int(payload.get("sub"))
    jti = payload.get("jti")
    
    # Check blacklist
    blacklist = TokenBlacklist(request.app.state.redis)
    if jti and await blacklist.is_blacklisted(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )
    
    # Verify user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Blacklist old token
    if jti:
        expires = datetime.utcfromtimestamp(payload.get("exp"))
        await blacklist.blacklist_token(jti, expires)
    
    # Issue new tokens
    new_access = create_access_token(
        subject=str(user.id),
        additional_claims={
            "email": user.email,
            "role": user.role.value,
        }
    )
    new_refresh, refresh_expires = create_refresh_token(subject=str(user.id))
    
    return TokenResponse(data={
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    })


@router.post("/logout", response_model=TokenResponse)
async def logout(
    request: Request,
    current_user: CurrentUser,
    credentials: Annotated[HTTPBearer, Depends(security)]
):
    """
    Logout - blacklist current access token
    """
    token = credentials.credentials
    jti = get_token_jti(token)
    
    if jti:
        payload = decode_token(token)
        if payload:
            expires = datetime.utcfromtimestamp(payload.get("exp"))
            blacklist = TokenBlacklist(request.app.state.redis)
            await blacklist.blacklist_token(jti, expires)
    
    # Audit log
    # TODO: Add audit log for logout
    
    return TokenResponse(data={"message": "Logged out successfully"})


@router.get("/me", response_model=TokenResponse)
async def get_me(current_user: CurrentUser):
    """Get current user profile"""
    return TokenResponse(data={
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value,
        "kyc_status": current_user.kyc_status.value,
        "is_2fa_enabled": current_user.is_2fa_enabled,
        "profile": {
            "full_name": current_user.profile.full_name if current_user.profile else None,
            "country": current_user.profile.country if current_user.profile else None,
            "accreditation_status": current_user.profile.accreditation_status.value if current_user.profile else None,
        }
    })


@router.post("/enable-2fa", response_model=TokenResponse)
async def enable_2fa(
    request: Request,
    req: Enable2FARequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db)
):
    """
    Enable 2FA for user
    Returns QR code URI for setup
    """
    if current_user.is_2fa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled",
        )
    
    # Generate TOTP secret
    secret = generate_totp_secret()
    
    # Verify user can set it up (must provide code)
    if not verify_totp(req.totp_code, secret):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid 2FA code. Please scan QR and try again.",
        )
    
    # Save secret
    current_user.totp_secret = secret
    current_user.is_2fa_enabled = True
    
    # Audit log
    audit = AuditLog(
        actor_id=current_user.id,
        action=AuditAction.TWO_FA_ENABLED,
        target_type="user",
        target_id=current_user.id,
        ip_address=request.client.host if request.client else None,
    )
    db.add(audit)
    
    await db.commit()
    
    return TokenResponse(data={"message": "2FA enabled successfully"})


@router.post("/2fa/setup", response_model=TokenResponse)
async def setup_2fa(current_user: CurrentUser):
    """Get QR code for 2FA setup (before enabling)"""
    if current_user.is_2fa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled",
        )
    
    # Generate temporary secret
    secret = generate_totp_secret()
    
    # Generate QR URI
    uri = get_totp_provisioning_uri(secret, current_user.email)
    
    return TokenResponse(data={
        "secret": secret,
        "qr_uri": uri,
        "manual_entry": secret,
    })


@router.post("/forgot-password", response_model=TokenResponse)
async def forgot_password(
    request: Request,
    req: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Request password reset - sends email with token"""
    result = await db.execute(
        select(User).where(User.email == req.email.lower())
    )
    user = result.scalar_one_or_none()
    
    # Always return success (don't reveal if email exists)
    if user:
        token = generate_password_reset_token()
        user.password_reset_token = token
        user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
        await db.commit()
        
        # Send email (background)
        # background_tasks.add_task(send_password_reset_email, user.email, token)
    
    return TokenResponse(data={
        "message": "If an account exists with this email, you will receive a password reset link."
    })


@router.post("/reset-password", response_model=TokenResponse)
async def reset_password(
    request: Request,
    req: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """Reset password with token from email"""
    result = await db.execute(
        select(User).where(
            User.password_reset_token == req.token,
            User.password_reset_expires > datetime.utcnow()
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )
    
    # Update password
    user.password_hash = get_password_hash(req.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    
    # Audit log
    audit = AuditLog(
        actor_id=user.id,
        action=AuditAction.PASSWORD_RESET_COMPLETED,
        target_type="user",
        target_id=user.id,
        ip_address=request.client.host if request.client else None,
    )
    db.add(audit)
    
    await db.commit()
    
    return TokenResponse(data={"message": "Password reset successfully"})
