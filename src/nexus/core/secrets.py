import os
from typing import Optional
from dotenv import load_dotenv
from .enterprise_logger import get_enterprise_logger

logger = get_enterprise_logger("secrets_manager")

class SecretsManager:
    """
    Standardized secret management for API keys and credentials.
    Loads from environment variables or .env file.
    """
    def __init__(self):
        load_dotenv()
        self.logger = logger

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieves a secret from the environment."""
        value = os.getenv(key, default)
        if not value:
            self.logger.warn(f"Secret {key} not found in environment.")
        return value

    def get_alpaca_creds(self) -> dict:
        """Helper to get Alpaca-specific credentials."""
        return {
            "key": self.get_secret("ALPACA_API_KEY"),
            "secret": self.get_secret("ALPACA_SECRET_KEY"),
            "base_url": self.get_secret("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        }

# Global singleton
secrets = SecretsManager()
