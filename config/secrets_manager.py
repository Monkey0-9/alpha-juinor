"""
config/secrets_manager.py

Centralized Secrets Management.
Abstraction layer for HashiCorp Vault / AWS Secrets Manager.
Enforces "No API Keys in Code" policy.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("SECRETS")

class SecretManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecretManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.vault_client = None
        # Future: Initialize hvac for HashiVault
        # Future: Initialize boto3 for AWS Secrets
        self.env_fallback = True # Allowed for now, disable in PROD

    def get_secret(self, key_name: str) -> Optional[str]:
        """
        Retrieve a secret by name.
        Order:
        1. Vault (if active)
        2. Environment Variable (Fallback)
        """
        # 1. Try Vault
        if self.vault_client:
            # implementation placeholder
            pass

        # 2. Environment Fallback
        if self.env_fallback:
            val = os.getenv(key_name)
            if val:
                return val

        logger.error(f"[SECRETS] Secret '{key_name}' not found in any store.")
        return None

# Global instance
secrets = SecretManager()
