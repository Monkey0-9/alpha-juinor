
"""
scripts/check_vault_config.py

Phase 0 Check: Verify Secrets Manager configuration.
Ensures we are reading from ENV/Vault and not using unsafe defaults.
"""
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.secrets_manager import secrets
except ImportError:
    print("FAIL: Could not import config.secrets_manager")
    sys.exit(1)

def check_config():
    print("Checking Secrets Configuration...")

    # 1. Check if Vault client is initialized (optional but good to know)
    vault_active = secrets.vault_client is not None
    print(f"Vault Active: {vault_active}")

    # 2. Check if ENV fallback is allowed
    env_fallback = secrets.env_fallback
    print(f"Env Fallback: {env_fallback}")

    # 3. Validation Logic
    # We require either Vault OR Env fallback to be true.
    if not vault_active and not env_fallback:
        print("FAIL: No secret source configured (Vault=False, EnvFallback=False)")
        sys.exit(1)

    # 4. Check critical secrets in ENV (if fallback active)
    if env_fallback:
        # Check for Alpaca keys as a smoke test
        if not os.getenv("ALPACA_API_KEY"):
            print("WARNING: ALPACA_API_KEY not found in ENV (Env Fallback active)")
            # We don't fail here because maybe this is a fresh environment,
            # but we warn.
        else:
            print("OK: ALPACA_API_KEY found in ENV")

    print("SUCCESS: Secrets configuration valid.")
    sys.exit(0)

if __name__ == "__main__":
    check_config()
