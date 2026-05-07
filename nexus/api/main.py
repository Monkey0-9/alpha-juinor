from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from nexus.api.alpaca_router import router as alpaca_router
from nexus.api.monitor_router import router as monitor_router
from nexus.execution.alpaca import get_client
from nexus.utils.config import Config

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Nexus API Backend...")
    valid, missing = Config.validate()
    if not valid:
        logger.warning("Alpaca credentials are not configured. Execution routes will remain disabled until credentials are provided.")

    client = get_client()
    if client.enabled:
        try:
            acc = await client.get_account()
            status = acc.get("status")
            if status == "ACTIVE":
                logger.info("Alpaca execution link established.")
            else:
                error_msg = acc.get("error", "Unknown Error")
                logger.warning(f"Alpaca account returned status: {status} | Error: {error_msg}")
                if status == "UNAUTHORIZED":
                    logger.error("CRITICAL: Alpaca API Keys are invalid or unauthorized. Please check your .env file.")
        except Exception as exc:
            logger.warning(f"Unable to verify Alpaca account during startup: {exc}")

    yield
    await get_client().close()
    logger.info("Nexus API Backend shutdown complete.")

app = FastAPI(
    title="Nexus Institutional API",
    description="Unified API for market data, execution, and market intelligence.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(alpaca_router)
app.include_router(monitor_router)

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
