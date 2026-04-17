import logging
import threading
from typing import Optional, Dict, Any
from .production_config import config_manager, ProductionConfig
from .enterprise_logger import get_enterprise_logger, EnterpriseLogger

class NexusContext:
    """
    Global Context for the Nexus Trading Engine.
    Handles configuration, logging, and shared engine state.
    Implemented as a thread-safe singleton.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.config: ProductionConfig = config_manager.get_config()
        self.logger: EnterpriseLogger = get_enterprise_logger("nexus_core")
        self.engine_state: Dict[str, Any] = {
            "is_running": False,
            "mode": self.config.environment.value,
            "start_time": None
        }
        self._initialized = True
        self.logger.info("NexusContext initialized successfully")

    def initialize(self, config_path: str):
        """Re-initializes the global context with a specific config."""
        self.config = config_manager.get_config(config_path)
        self.engine_state["mode"] = self.config.environment.value
        self.logger.info(f"NexusContext re-initialized with config from {config_path}")

    def get_logger(self, name: str) -> EnterpriseLogger:
        return get_enterprise_logger(name)

    def set_running(self, running: bool):
        from datetime import datetime
        self.engine_state["is_running"] = running
        if running:
            self.engine_state["start_time"] = datetime.utcnow().isoformat()
            self.logger.info(f"Nexus Engine started in {self.engine_state['mode']} mode")
        else:
            self.logger.info("Nexus Engine stopped")

# Global accessor
engine_context = NexusContext()
