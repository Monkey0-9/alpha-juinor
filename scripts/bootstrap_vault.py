
import os
import logging
import sys
from dotenv import load_dotenv

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.secrets_vault import VaultClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VaultBootstrap")

load_dotenv()

def bootstrap():
    """
    Initialize Vault with secrets from .env
    Prerequisites: Vault server running (docker run -p 8200:8200 -e 'VAULT_DEV_ROOT_TOKEN_ID=root' hashicorp/vault)
    """
    vault = VaultClient()

    if not vault.client or not vault.client.is_authenticated():
        logger.error("Vault client not authenticated. Please check VAULT_ADDR and VAULT_TOKEN.")
        logger.info("Tip: Run `docker run -d --name vault -p 8200:8200 -e 'VAULT_DEV_ROOT_TOKEN_ID=root' hashicorp/vault`")
        return

    secrets_to_migrate = {
        "quant-fund/api-keys": {
            "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY"),
            "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
            "DATA_API_KEY": os.getenv("DATA_API_KEY"),
            "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY"),
            "FRED_API_KEY": os.getenv("FRED_API_KEY")
        },
        "quant-fund/database": {
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "DB_CONNECTION_STRING": os.getenv("DB_CONNECTION_STRING")
        }
    }

    count = 0
    for path, secrets in secrets_to_migrate.items():
        # filter None values
        valid_secrets = {k: v for k, v in secrets.items() if v is not None}
        if not valid_secrets:
            continue

        try:
            vault.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=valid_secrets,
                mount_point=vault.mount_point
            )
            logger.info(f"Wrote {len(valid_secrets)} secrets to {path}")
            count += 1
        except Exception as e:
            logger.error(f"Failed to write to {path}: {e}")

    logger.info(f"Bootstrap complete. Migrated {count} paths.")

if __name__ == "__main__":
    bootstrap()
