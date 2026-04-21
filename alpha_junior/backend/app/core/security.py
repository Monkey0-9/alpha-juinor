"""
Security utilities for Alpha Junior
JWT, password hashing, 2FA, token management
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import hashlib
import secrets
import base64

from passlib.context import CryptContext
import pyotp
import jwt
from jwt import PyJWTError

from app.core.config import settings

# Password hashing with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password with bcrypt (cost factor 12)"""
    return pwd_context.hash(password)


def generate_totp_secret() -> str:
    """Generate new TOTP secret for 2FA setup"""
    return pyotp.random_base32()


def verify_totp(token: str, secret: str) -> bool:
    """Verify TOTP code"""
    totp = pyotp.TOTP(secret)
    # Allow 1 time step drift (30 seconds before/after)
    return totp.verify(token, valid_window=1)


def get_totp_provisioning_uri(secret: str, email: str) -> str:
    """Generate QR code URI for 2FA setup"""
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(
        name=email,
        issuer_name=settings.TOTP_ISSUER_NAME
    )


def generate_email_verification_token() -> str:
    """Generate secure random token for email verification"""
    return secrets.token_urlsafe(32)


def generate_password_reset_token() -> str:
    """Generate secure random token for password reset"""
    return secrets.token_urlsafe(32)


def generate_refresh_token() -> str:
    """Generate secure refresh token"""
    return secrets.token_urlsafe(64)


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create JWT access token with RS256 (asymmetric)
    
    Args:
        subject: User ID (sub claim)
        expires_delta: Token lifetime (default: 15 minutes)
        additional_claims: Extra JWT claims (role, permissions, etc.)
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    # Sign with RSA private key
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_PRIVATE_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt


def create_refresh_token(subject: str) -> Tuple[str, datetime]:
    """
    Create JWT refresh token
    
    Returns:
        (token, expiration_datetime)
    """
    expires = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": str(subject),
        "exp": expires,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "jti": secrets.token_hex(16),  # Unique token ID for blacklist
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_PRIVATE_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt, expires


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and verify JWT token
    
    Returns:
        Payload dict if valid, None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_PUBLIC_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except PyJWTError:
        return None


def get_token_jti(token: str) -> Optional[str]:
    """Extract JWT ID (jti) from token for blacklist"""
    payload = decode_token(token)
    if payload:
        return payload.get("jti")
    return None


def generate_secure_id() -> str:
    """Generate cryptographically secure random ID"""
    return secrets.token_urlsafe(16)


def hash_ip_address(ip: str) -> str:
    """Hash IP address for privacy"""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def mask_email(email: str) -> str:
    """Mask email for display (e.g., j***@example.com)"""
    if "@" not in email:
        return "***"
    
    local, domain = email.split("@")
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
    
    return f"{masked_local}@{domain}"


class TokenBlacklist:
    """
    Redis-backed token blacklist for logout/revocation
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.key_prefix = "token_blacklist:"
    
    async def blacklist_token(self, jti: str, expires_at: datetime) -> None:
        """Add token JTI to blacklist until expiration"""
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        if ttl > 0:
            await self.redis.setex(
                f"{self.key_prefix}{jti}",
                ttl,
                "revoked"
            )
    
    async def is_blacklisted(self, jti: str) -> bool:
        """Check if token JTI is blacklisted"""
        exists = await self.redis.exists(f"{self.key_prefix}{jti}")
        return bool(exists)


class RateLimiter:
    """
    Redis-backed rate limiter
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def is_allowed(
        self,
        key: str,
        max_requests: int = settings.RATE_LIMIT_REQUESTS,
        window: int = settings.RATE_LIMIT_WINDOW
    ) -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit
        
        Returns:
            (allowed: bool, remaining: int)
        """
        current = await self.redis.get(key)
        
        if current is None:
            # First request in window
            await self.redis.setex(key, window, 1)
            return True, max_requests - 1
        
        current_count = int(current)
        
        if current_count >= max_requests:
            return False, 0
        
        # Increment counter
        await self.redis.incr(key)
        return True, max_requests - current_count - 1
    
    async def record_login_attempt(self, ip: str) -> int:
        """Record failed login attempt, return count"""
        key = f"login_attempts:{ip}"
        current = await self.redis.get(key)
        
        if current is None:
            await self.redis.setex(key, settings.LOGIN_LOCKOUT_MINUTES * 60, 1)
            return 1
        
        new_count = await self.redis.incr(key)
        return int(new_count)
    
    async def is_login_locked(self, ip: str) -> bool:
        """Check if IP is locked out due to failed logins"""
        key = f"login_attempts:{ip}"
        attempts = await self.redis.get(key)
        
        if attempts is None:
            return False
        
        return int(attempts) >= settings.MAX_LOGIN_ATTEMPTS
    
    async def clear_login_attempts(self, ip: str) -> None:
        """Clear failed login attempts (on successful login)"""
        await self.redis.delete(f"login_attempts:{ip}")


# Rate limit key generators
def get_rate_limit_key_ip(ip: str) -> str:
    """Generate rate limit key for IP"""
    return f"rate_limit:ip:{ip}"


def get_rate_limit_key_user(user_id: int) -> str:
    """Generate rate limit key for user"""
    return f"rate_limit:user:{user_id}"
