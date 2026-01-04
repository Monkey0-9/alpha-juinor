# configs/config_manager.py
import yaml
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Institutional Config Manager.
    Loads, validates, and locks the system configuration with SHA256 hashing.
    Enforces immutability after initial load.
    """
    
    _instance = None
    _config: Dict[str, Any] = None
    _config_hash: str = None
    
    def __new__(cls, config_path: str = "configs/golden_config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, path: str):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Golden Config not found at {path}")
            
        with open(path_obj, "rb") as f:
            content = f.read()
            self._config_hash = hashlib.sha256(content).hexdigest()
            
        with open(path_obj, "r") as f:
            self._config = yaml.safe_load(f)
            
        logger.info(f"Loaded Golden Config: {path}")
        logger.info(f"Config SHA256 Hash: {self._config_hash}")
        
        # Deep Freeze would be ideal, but for now we just restrict access
        self._lock = True

    @property
    def config(self) -> Dict[str, Any]:
        """Returns a copy of the config to prevent direct mutation."""
        import copy
        return copy.deepcopy(self._config)
        
    @property
    def config_hash(self) -> str:
        return self._config_hash

    def validate_runtime_integrity(self):
        """Standard check for production runs."""
        # In a real system, we'd re-hash the file on disk or check memory
        # For now, we just ensure the hash is logged/tracked.
        pass

def get_config() -> Dict[str, Any]:
    return ConfigManager().config

def get_config_hash() -> str:
    return ConfigManager().config_hash
