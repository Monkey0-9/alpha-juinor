import logging
import asyncio
import time
from typing import Dict, Any, Optional
from nexus.utils.notifications import notifier

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors platform vital signs and broadcasts critical alerts."""

    def __init__(self):
        self.stats = {
            "market": {"status": "healthy", "details": "N/A"},
            "risk": {"status": "healthy", "details": "N/A"},
            "backend": {"status": "healthy", "details": "N/A"},
            "market_session": {"status": "healthy", "details": "N/A"},
        }
        self.last_heartbeat = 0.0

    def record(
        self, component: str, healthy: bool, details: Any = "N/A"
    ):
        status = "healthy" if healthy else "failed"
        self.stats[component] = {"status": status, "details": details}

        if not healthy:
            logger.critical(
                f"HEALTH ALERT: {component} has failed. "
                f"Details: {details}"
            )
            # Broadcast to Telegram/Discord
            asyncio.create_task(
                notifier.notify(
                    f"CRITICAL HEALTH ALERT: {component} failure. "
                    f"{details}",
                    level="CRITICAL",
                )
            )
        else:
            logger.info(
                f"HealthMonitor [{component}] status={status} "
                f"details={details}"
            )

    def heartbeat(self):
        self.last_heartbeat = time.time()
