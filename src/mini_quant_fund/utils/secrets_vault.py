
import os
import logging
from typing import Optional, Dict, Any
import hvac
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("SecretsVault")

class VaultClient:
    """
    Singleton client for HashiCorp Vault.
    Provides fallback to environment variables if Vault is unreachable or unconfigured.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VaultClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.url = os.getenv("VAULT_ADDR", "http://localhost:8200")
        self.token = os.getenv("VAULT_TOKEN", "root") # Default dev token
        self.mount_point = os.getenv("VAULT_MOUNT_POINT", "secret")
        self.client = None
        self.enabled = os.getenv("VAULT_ENABLED", "false").lower() == "true"

        if self.enabled:
            try:
                self.client = hvac.Client(url=self.url, token=self.token)
                if self.client.is_authenticated():
                    logger.info(f"Connected to Vault at {self.url}")
                else:
                    logger.warning("Vault authentication failed. Falling back to ENV.")
                    self.client = None
            except Exception as e:
                logger.warning(f"Failed to connect to Vault: {e}. Falling back to ENV.")
                self.client = None

        self._initialized = True

    def get_secret(self, path: str, key: str = None) -> Optional[Any]:
        """
        Get a secret from Vault, or fallback to ENV.

        Args:
            path: Path in Vault (e.g., 'quant-fund/api-keys')
            key: Specific key in the secret dict (e.g., 'ALPACA_API_KEY')
                 If None, returns the whole dictionary.

        Returns:
            Secret value or None.
        """
        # 1. Try Vault
        if self.client:
            try:
                # Read from KV v2 mount
                read_response = self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                    mount_point=self.mount_point
                )
                data = read_response['data']['data']

                if key:
                    if key in data:
                        return data[key]
                else:
                    return data

            except Exception as e:
                logger.debug(f"Vault lookup failed for {path}/{key}: {e}")

        # 2. Fallback to ENV (only if key provided)
        if key:
            # Map key to ENV var naming convention if needed, usually same
            env_val = os.getenv(key)
            if env_val:
                logger.debug(f"Loaded {key} from ENV")
                return env_val

        return None

# Global Instance
vault = VaultClient()
