"""
Local development configuration - No Docker needed
Uses SQLite instead of PostgreSQL, in-memory cache instead of Redis
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Alpha Junior"
    API_V1_STR: str = "/api/v1"
    
    # SQLite for local development (no Docker needed!)
    DATABASE_URL: str = "sqlite+aiosqlite:///./alpha_junior.db"
    
    @property
    def async_database_url(self) -> str:
        return self.DATABASE_URL

    # Simple in-memory for local (no Redis needed)
    REDIS_URL: str = "memory://"
    
    # Auth - Simple for local dev
    SECRET_KEY: str = "local-development-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"  # Simpler than RS256 for local
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # 2FA
    TOTP_ISSUER_NAME: str = "Alpha Junior"
    
    # Security
    PASSWORD_BCRYPT_ROUNDS: int = 12
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15
    
    # File uploads (local storage)
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    
    # Email (console output for local dev)
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 1025
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "noreply@alphajunior.local"
    SMTP_FROM_NAME: str = "Alpha Junior"
    
    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 1000  # Higher for local dev
    RATE_LIMIT_WINDOW: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # External APIs (optional - leave empty if you don't have keys)
    ALPHA_VANTAGE_API_KEY: str = "demo"
    NEWS_API_KEY: str = ""
    FRED_API_KEY: str = ""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
