from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict
import logging
import os
from nexus.api.alpaca_router import router as alpaca_router
from nexus.api.monitor_router import router as monitor_router
from nexus.execution.alpaca import get_client
from nexus.utils.config import Config

logger = logging.getLogger(__name__)

# API key loaded once at import time
_API_KEY = os.getenv("NEXUS_API_KEY", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Nexus API Backend...")
    valid, missing = Config.validate()
    if not valid:
        logger.warning(
            "Alpaca credentials missing. "
            "Execution routes disabled until configured."
        )

    client = get_client()
    if client.enabled:
        try:
            acc = await client.get_account()
            status = acc.get("status")
            if status == "ACTIVE":
                logger.info("Alpaca execution link established.")
            else:
                error_msg = acc.get("error", "Unknown Error")
                logger.warning(
                    f"Alpaca status: {status} | Error: {error_msg}"
                )
                if status == "UNAUTHORIZED":
                    logger.error(
                        "CRITICAL: Alpaca API Keys invalid. "
                        "Check .env file."
                    )
        except Exception as exc:
            logger.warning(
                f"Unable to verify Alpaca account: {exc}"
            )

    yield
    await get_client().close()
    logger.info("Nexus API Backend shutdown complete.")


app = FastAPI(
    title="Nexus Institutional API",
    description="Unified API for market data and execution.",
    version="2.0.0",
    lifespan=lifespan,
)


# --- API Key Authentication Middleware ---
@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    """Verify X-API-Key header on mutation endpoints."""
    # Skip auth for health checks and GET-only read endpoints
    logger.info(f"API Request: {request.method} {request.url.path}")
    safe_paths = {"/docs", "/openapi.json", "/api/alpaca/health"}
    if request.url.path in safe_paths:
        logger.info(f"Safe path detected: {request.url.path}")
        return await call_next(request)

    # If no API key is configured, skip auth (dev mode)
    if not _API_KEY:
        return await call_next(request)

    # Enforce auth on POST, DELETE, PUT, PATCH
    if request.method in {"POST", "DELETE", "PUT", "PATCH"}:
        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != _API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

    return await call_next(request)


# --- CORS — Restricted to Streamlit dashboard ---
allowed_origins = [
    f"http://localhost:{Config.STREAMLIT_PORT}",
    f"http://127.0.0.1:{Config.STREAMLIT_PORT}",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "Nexus API",
        "version": "2.0.0"
    }

app.include_router(alpaca_router)
app.include_router(monitor_router)

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
