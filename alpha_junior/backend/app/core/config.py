from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Alpha Junior"
    API_V1_STR: str = "/api/v1"
    
    # DATABASE
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "alpha_junior"
    DATABASE_URL: Optional[str] = None

    @property
    def async_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"

    # REDIS
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # AUTH - JWT (RS256 for asymmetric security)
    JWT_PRIVATE_KEY: str = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...REPLACE_WITH_REAL_KEY_IN_PRODUCTION...
-----END RSA PRIVATE KEY-----"""
    JWT_PUBLIC_KEY: str = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...REPLACE_WITH_REAL_KEY...
-----END PUBLIC KEY-----"""
    JWT_ALGORITHM: str = "RS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # 2FA
    TOTP_ISSUER_NAME: str = "Alpha Junior"
    
    # SECURITY
    PASSWORD_BCRYPT_ROUNDS: int = 12
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15
    
    # S3 STORAGE
    S3_ENDPOINT_URL: Optional[str] = None  # For R2/MinIO, None for AWS
    S3_BUCKET_NAME: str = "alpha-junior-documents"
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_REGION: str = "us-east-1"
    S3_PRESIGNED_URL_EXPIRY: int = 3600  # 1 hour
    MAX_UPLOAD_SIZE_MB: int = 50
    
    # EMAIL (SMTP)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "noreply@alphajunior.com"
    SMTP_FROM_NAME: str = "Alpha Junior"
    
    # FRONTEND
    FRONTEND_URL: str = "http://localhost:3000"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # CELERY
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    
    @property
    def celery_broker(self) -> str:
        return self.CELERY_BROKER_URL or self.redis_url
    
    # RATE LIMITING
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # LOGGING
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    
    # KYC
    KYC_VERIFICATION_PROVIDER: Optional[str] = None  # "sumsub", "onfido", "jumio"
    KYC_AUTO_APPROVE_THRESHOLD: float = 0.95  # AI confidence threshold
    
    # FUNDS
    DEFAULT_CURRENCY: str = "USD"
    MAX_FUND_NAME_LENGTH: int = 255
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
