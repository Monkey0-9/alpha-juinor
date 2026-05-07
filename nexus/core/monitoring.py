import logging
import time
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)

class HealthMonitor:
    """Tracks system health and triggers graceful pauses or alerts."""

    def __init__(self, service_name: str = "Nexus", max_failures: int = 5):
        self.service_name = service_name
        self.max_failures = max_failures
        self.failure_counts: Dict[str, int] = {}
        self.last_updated = time.time()
        self.alert_callbacks: list[Callable[[str, str], None]] = []

    def add_alert_callback(self, callback: Callable[[str, str], None]):
        self.alert_callbacks.append(callback)

    def record(self, component: str, healthy: bool, details: Optional[str] = None) -> None:
        self.last_updated = time.time()
        status = "healthy" if healthy else "unhealthy"
        logger.info(f"HealthMonitor [{component}] status={status} details={details or 'n/a'}")

        if not healthy:
            self.failure_counts[component] = self.failure_counts.get(component, 0) + 1
            if self.is_critical(component):
                self.alert(component, details)
        else:
            self.failure_counts[component] = 0

    def is_critical(self, component: str) -> bool:
        return self.failure_counts.get(component, 0) >= self.max_failures

    def should_pause(self, component: str) -> bool:
        return self.is_critical(component)

    def alert(self, component: str, details: Optional[str] = None) -> None:
        message = f"[ALERT] {self.service_name} component '{component}' failed {self.failure_counts.get(component, 0)} times. {details or ''}"
        logger.warning(message)
        for callback in self.alert_callbacks:
            try:
                callback(component, message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def heartbeat(self) -> None:
        logger.debug(f"HealthMonitor heartbeat at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.last_updated))}")
