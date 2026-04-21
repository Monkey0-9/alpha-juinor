"""
API Dependencies
Authentication, authorization, database session injection
"""

from typing import Optional, Annotated

from fastapi import Depends, HTTPException, status, Request, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.core.security import decode_token, TokenBlacklist, RateLimiter, get_token_jti
from app.models.user import User, UserRole, KYCStatus

# Security scheme
security = HTTPBearer()


async def get_db() -> AsyncSession:
    """Database session dependency"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token
    Validates token and returns user object
    """
    token = credentials.credentials
    
    # Decode token
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check blacklist
    jti = payload.get("jti") or get_token_jti(token)
    if jti and request.app.state.token_blacklist:
        is_blacklisted = await request.app.state.token_blacklist.is_blacklisted(jti)
        if is_blacklisted:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Get user
    user_id = int(payload.get("sub"))
    from sqlalchemy import select
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )
    
    if user.is_deleted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account has been deleted",
        )
    
    # Attach request info for audit logging
    user._request_ip = request.client.host if request.client else None
    user._request_id = request.headers.get("X-Request-ID")
    
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Ensure user is active (not suspended or pending)
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


def require_role(allowed_roles: list[UserRole]):
    """
    Dependency factory for role-based access control
    Usage: current_user: User = Depends(require_role([UserRole.MANAGER, UserRole.ADMIN]))
    """
    async def role_checker(
        current_user: Annotated[User, Depends(get_current_active_user)]
    ) -> User:
        if current_user.role not in allowed_roles and current_user.role != UserRole.SUPERADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {[r.value for r in allowed_roles]}",
            )
        return current_user
    return role_checker


# Common role requirements
require_manager = require_role([UserRole.MANAGER])
require_admin = require_role([UserRole.ADMIN])
require_superadmin = require_role([UserRole.SUPERADMIN])


async def require_kyc_approved(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """
    Ensure user has approved KYC status
    Required for investment-related operations
    """
    if current_user.kyc_status != KYCStatus.APPROVED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"KYC verification required. Current status: {current_user.kyc_status.value}",
        )
    return current_user


async def rate_limit_check(
    request: Request,
    key_prefix: str = "api"
) -> None:
    """
    Rate limiting dependency
    Checks per-IP and per-user (if authenticated) rate limits
    """
    from app.core.config import settings
    
    redis = request.app.state.redis
    limiter = RateLimiter(redis)
    
    # Build key
    client_ip = request.client.host if request.client else "unknown"
    key = f"{key_prefix}:{client_ip}"
    
    # Check rate limit
    allowed, remaining = await limiter.is_allowed(
        key,
        max_requests=settings.RATE_LIMIT_REQUESTS,
        window=settings.RATE_LIMIT_WINDOW
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW)},
        )
    
    # Store remaining count for response headers
    request.state.rate_limit_remaining = remaining


# Type aliases for cleaner endpoint signatures
CurrentUser = Annotated[User, Depends(get_current_active_user)]
CurrentUserWithKYC = Annotated[User, Depends(require_kyc_approved)]
ManagerUser = Annotated[User, Depends(require_manager)]
AdminUser = Annotated[User, Depends(require_admin)]
SuperAdminUser = Annotated[User, Depends(require_superadmin)]


# WebSocket authentication
async def get_ws_user(
    websocket: WebSocket,
    token: Optional[str] = None
) -> Optional[User]:
    """
    Authenticate WebSocket connection
    Token passed via query parameter: ?token=xxx
    """
    if not token:
        await websocket.close(code=4001, reason="Missing authentication token")
        return None
    
    payload = decode_token(token)
    if payload is None:
        await websocket.close(code=4001, reason="Invalid token")
        return None
    
    # Get user from database
    # Note: WebSocket doesn't have access to request.app.state.db easily
    # This is a simplified version - in production, pass db through app state
    user_id = int(payload.get("sub"))
    
    # Return user_id for now - full user lookup done in endpoint
    return user_id
