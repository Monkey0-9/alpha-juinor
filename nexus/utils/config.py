import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration for the Nexus platform."""

    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
    ALPACA_PAPER = (
        os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
    )
    ALLOW_SIMULATION_FALLBACK = (
        os.getenv("NEXUS_ALLOW_SIMULATION_FALLBACK", "false").lower() == "true"
    )

    UNIVERSE_RESCAN_INTERVAL = int(
        os.getenv("NEXUS_RESCAN_INTERVAL", "14400")
    )

    API_HOST = os.getenv("NEXUS_API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("NEXUS_API_PORT", "8001"))
    STREAMLIT_PORT = int(
        os.getenv("NEXUS_STREAMLIT_PORT", "8501")
    )
    BACKEND_URL = os.getenv(
        "NEXUS_BACKEND_URL",
        f"http://127.0.0.1:{API_PORT}",
    )

    # Authentication
    API_KEY = os.getenv("NEXUS_API_KEY", "")

    MAX_POSITION_SIZE = float(
        os.getenv("NEXUS_MAX_POSITION_SIZE", "0.05")
    )
    MAX_DRAWDOWN = float(
        os.getenv("NEXUS_MAX_DRAWDOWN", "0.15")
    )
    MAX_OPEN_POSITIONS = int(
        os.getenv("NEXUS_MAX_OPEN_POSITIONS", "50")
    )
    MAX_DAILY_TRADES = int(
        os.getenv("NEXUS_MAX_DAILY_TRADES", "100")
    )
    MIN_ORDER_USD = float(
        os.getenv("NEXUS_MIN_ORDER_USD", "50")
    )
    MAX_UNIVERSE_ASSETS = int(
        os.getenv("NEXUS_MAX_UNIVERSE_ASSETS", "1000")
    )
    CANDIDATE_POOL_SIZE = int(
        os.getenv("NEXUS_CANDIDATE_POOL_SIZE", "100")
    )
    TRADE_ALL_ASSETS = os.getenv("NEXUS_TRADE_ALL", "false").lower() == "true"
    MIN_HOLD_CYCLES = int(
        os.getenv("NEXUS_MIN_HOLD_CYCLES", "3")
    )
    HEARTBEAT_INTERVAL = int(
        os.getenv("NEXUS_HEARTBEAT_INTERVAL", "60")
    )
    MAX_RESTARTS = int(os.getenv("NEXUS_MAX_RESTARTS", "5"))

    # Institutional Trading Parameters
    TAKE_PROFIT_THRESHOLD = float(
        os.getenv("NEXUS_TAKE_PROFIT", "0.08")
    )
    STOP_LOSS_THRESHOLD = float(
        os.getenv("NEXUS_STOP_LOSS", "-0.04")
    )
    FALLBACK_EQUITY = float(
        os.getenv("NEXUS_FALLBACK_EQUITY", "1000000.0")
    )

    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """Validate that essential configuration is present."""
        missing = []
        if not cls.ALPACA_API_KEY:
            missing.append("ALPACA_API_KEY")
        if not cls.ALPACA_API_SECRET:
            missing.append("ALPACA_API_SECRET")
        return len(missing) == 0, missing

    @classmethod
    def ensure_ready(cls) -> None:
        valid, missing = cls.validate()
        if not valid:
            raise RuntimeError(
                f"Missing env vars: {', '.join(missing)}"
            )

    @classmethod
    def get_backend_url(cls) -> str:
        return cls.BACKEND_URL
