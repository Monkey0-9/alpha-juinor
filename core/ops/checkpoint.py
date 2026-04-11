
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("CHECKPOINT")

class CheckpointManager:
    """
    Handles Disaster Recovery by persisting system state.
    Bridges the gap to 'High Availability' and 'Disaster Recovery'.
    """
    def __init__(self, storage_dir: str = "runtime/checkpoints"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"CheckpointManager initialized at {storage_dir}")

    def save_state(self, component_name: str, state: Dict[str, Any]):
        """Persist state to a JSON file (simulating durable storage)."""
        filename = f"{component_name}_state.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        payload = {
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "data": state
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(payload, f, indent=2)
            logger.debug(f"Saved checkpoint for {component_name}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint for {component_name}: {e}")

    def load_state(self, component_name: str) -> Dict[str, Any]:
        """Restore state from disk."""
        filename = f"{component_name}_state.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        if not os.path.exists(filepath):
            return {}
            
        try:
            with open(filepath, 'r') as f:
                payload = json.load(f)
                return payload.get("data", {})
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {component_name}: {e}")
            return {}

def get_checkpoint_manager() -> CheckpointManager:
    return CheckpointManager()
