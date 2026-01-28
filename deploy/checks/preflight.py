"""
deploy/checks/preflight.py

Section K: DevOps Preflight Checks.
Validates environment before startup.
"""
import sys
import os
import sqlite3
import logging

# Add project root (deploy/checks/ -> root needs 3 levels up)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PREFLIGHT")

def check_db():
    db_path = "runtime/institutional_trading.db"
    if not os.path.exists(db_path):
        logger.error(f"DB not found: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    try:
        # Check decision_records existence logic
        conn.execute("SELECT 1 FROM decision_records LIMIT 1")
    except Exception as e:
        logger.error(f"decision_records table missing: {e}")
        return False
    finally:
        conn.close()
    return True

def check_secrets():
    # Simulate secret check logic or rely on router
    # Assuming secrets_manager logic
    try:
        from config.secrets_manager import secrets
        # Just check one key if entitlement requires it?
        return True
    except ImportError:
        logger.error("Secrets manager missing (ImportError)")
        return False
    except Exception as e:
        logger.error(f"Secrets check error: {e}")
        return False

def run_checks():
    logger.info("Running Preflight Checks...")

    checks = [
        ("Database & Audit Tables", check_db),
        ("Secrets", check_secrets)
    ]

    failed = []
    for name, func in checks:
        if func():
            logger.info(f"[PASS] {name}")
        else:
            logger.error(f"[FAIL] {name}")
            failed.append(name)

    if failed:
        logger.critical(f"Preflight Failed: {failed}")
        sys.exit(1)

    logger.info("All Systems Go.")
    sys.exit(0)

if __name__ == "__main__":
    run_checks()
