"""
Alpha Junior - Local Development Version
Runs without Docker, uses SQLite, in-memory cache
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Use local config (SQLite, no Redis)
from app.core.config_local import settings
from app.api.v1.api import api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Fake Redis for local development (in-memory)
class FakeRedis:
    """In-memory Redis replacement for local development"""
    def __init__(self):
        self._data = {}
    
    async def get(self, key):
        return self._data.get(key)
    
    async def setex(self, key, seconds, value):
        self._data[key] = value
    
    async def set(self, key, value):
        self._data[key] = value
    
    async def incr(self, key):
        self._data[key] = self._data.get(key, 0) + 1
        return self._data[key]
    
    async def delete(self, key):
        if key in self._data:
            del self._data[key]
    
    async def exists(self, key):
        return 1 if key in self._data else 0
    
    async def close(self):
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting Alpha Junior (Local Development Mode)...")
    logger.info("Using SQLite database (no PostgreSQL needed)")
    logger.info("Using in-memory cache (no Redis needed)")
    
    # Initialize fake Redis
    app.state.redis = FakeRedis()
    
    # Initialize database tables
    from app.db.session_local import init_db
    await init_db()
    
    logger.info("✓ Alpha Junior is ready!")
    logger.info(f"API: http://localhost:8000{settings.API_V1_STR}")
    logger.info(f"Docs: http://localhost:8000{settings.API_V1_STR}/docs")
    
    yield
    
    logger.info("Shutting down...")
    await app.state.redis.close()


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Institutional Fund Management Platform - Local Dev Mode",
    version="1.0.0-local",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# CORS - allow all for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc)
            }
        }
    )

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME, "mode": "local"}

@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": "1.0.0-local",
        "docs": f"{settings.API_V1_STR}/docs",
        "mode": "local-development"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_local:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
