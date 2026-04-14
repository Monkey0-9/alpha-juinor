
import socket
import logging
import os
from typing import Dict, List

logger = logging.getLogger("INFRA_GUARD")

class InfrastructureGuard:
    """
    Ensures real-world microservices are alive before trading starts.
    Prevents 'Zombie Prototype' behavior by enforcing real connectivity.
    """
    def __init__(self):
        self.critical_services = {
            "Redis": {"host": os.getenv("REDIS_HOST", "localhost"), "port": 6379},
            "Postgres": {"host": os.getenv("POSTGRES_HOST", "localhost"), "port": 5432},
            "Kafka": {"host": os.getenv("KAFKA_HOST", "localhost"), "port": 9092}
        }

    def check_port(self, host: str, port: int) -> bool:
        """Check if a service port is actually open."""
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

    def verify_all(self) -> bool:
        """Verify all critical infrastructure."""
        logger.info("Starting Institutional Infrastructure Pre-Flight Check...")
        all_passed = True
        
        for name, cfg in self.critical_services.items():
            alive = self.check_port(cfg["host"], cfg["port"])
            if alive:
                logger.info(f"[PASS] {name} is ALIVE at {cfg['host']}:{cfg['port']}")
            else:
                logger.error(f"[FAIL] {name} is UNREACHABLE at {cfg['host']}:{cfg['port']}")
                all_passed = False
                
        return all_passed

def get_infra_guard() -> InfrastructureGuard:
    return InfrastructureGuard()
