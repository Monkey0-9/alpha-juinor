import os
import sys
import hashlib
from typing import Optional

# Add project root to path
sys.path.insert(0, os.getcwd())

from configs.config_manager import ConfigManager

def freeze_config(output_lock_file: Optional[str] = "configs/golden_config.lock"):
    """
    Load Golden Config, compute hash, and lock it.
    """
    print("❄️  Freezing Golden Configuration...")

    try:
        cm = ConfigManager()
        current_hash = cm.config_hash
        print(f"✅ Loaded Config. SHA256: {current_hash}")

        if output_lock_file:
            with open(output_lock_file, "w") as f:
                f.write(current_hash)
            print(f"🔒 Lock file written to: {output_lock_file}")

        return True
    except Exception as e:
        print(f"❌ Failed to freeze config: {e}")
        return False

if __name__ == "__main__":
    success = freeze_config()
    sys.exit(0 if success else 1)
