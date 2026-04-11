"""
API Key Rotation & Security Setup
===================================
Automates the process of rotating all previously-exposed
API keys and configuring the system for production.

Usage:
    python scripts/rotate_keys.py

This script:
1. Generates a fresh .env file with new key placeholders
2. Validates that no hardcoded secrets remain in source
3. Sets up git pre-commit hook for secret detection
4. Creates .gitignore entries for sensitive files
"""

import hashlib
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================
# 1. Key Inventory — All keys that were previously exposed
# ===========================================================

EXPOSED_KEYS = {
    "ALPACA_API_KEY": {
        "old_value": "***REDACTED***",
        "service": "Alpaca Markets",
        "rotation_url": "https://app.alpaca.markets/paper/dashboard/overview",
        "instructions": (
            "1. Go to https://app.alpaca.markets\n"
            "2. Navigate to Paper Trading > API Keys\n"
            "3. Click 'Regenerate' on both Key and Secret\n"
            "4. Copy new values into .env"
        ),
    },
    "ALPACA_SECRET_KEY": {
        "old_value": "***REDACTED***",
        "service": "Alpaca Markets",
        "rotation_url": "https://app.alpaca.markets/paper/dashboard/overview",
        "instructions": "Same as ALPACA_API_KEY — both rotate together.",
    },
    "POLYGON_API_KEY": {
        "old_value": "***REDACTED***",
        "service": "Polygon.io",
        "rotation_url": "https://polygon.io/dashboard/keys",
        "instructions": (
            "1. Go to https://polygon.io/dashboard/keys\n"
            "2. Click 'Regenerate API Key'\n"
            "3. Copy new key into .env"
        ),
    },
    "NEWS_API_KEY": {
        "old_value": "***REDACTED***",
        "service": "NewsAPI.org",
        "rotation_url": "https://newsapi.org/account",
        "instructions": (
            "1. Go to https://newsapi.org/account\n"
            "2. Generate a new API key\n"
            "3. Copy new key into .env"
        ),
    },
    "DEEPSEEK_API_KEY": {
        "old_value": "",  # Was in .env, may be exposed
        "service": "DeepSeek AI",
        "rotation_url": "https://platform.deepseek.com/api_keys",
        "instructions": (
            "1. Go to DeepSeek platform\n"
            "2. Regenerate API key\n"
            "3. Copy new key into .env"
        ),
    },
    "TELEGRAM_BOT_TOKEN": {
        "old_value": "",  # Was in .env
        "service": "Telegram Bot",
        "rotation_url": "https://t.me/BotFather",
        "instructions": (
            "1. Open Telegram, message @BotFather\n"
            "2. Send /revoke\n"
            "3. Select your bot\n"
            "4. Copy new token into .env"
        ),
    },
}


def print_header(title: str):
    """Print formatted section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def step_1_scan_for_secrets():
    """Scan codebase for any remaining hardcoded secrets."""
    print_header("STEP 1: Scanning for Remaining Secrets")

    # Known old key patterns to search for
    old_patterns = [v["old_value"] for v in EXPOSED_KEYS.values() if v["old_value"]]

    found_issues = []
    scan_extensions = {".py", ".yaml", ".yml", ".json", ".cfg", ".ini", ".toml"}
    skip_dirs = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        "logs",
        "data_lake",
        "db",
        "sqlite",
        ".tox",
        ".pytest_cache",
        "htmlcov",
        ".mypy_cache",
        "dist",
        "build",
        "mlruns",
        "checkpoints",
        "output",
    }
    max_files = 500
    scanned = 0
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix not in scan_extensions:
                continue

            # Skip large files
            try:
                if fpath.stat().st_size > 100_000:
                    continue
            except OSError:
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                for pattern in old_patterns:
                    if pattern in content:
                        rel = fpath.relative_to(PROJECT_ROOT)
                        found_issues.append((str(rel), pattern[:10] + "..."))
            except Exception:
                pass

    if found_issues:
        print("  [!] FOUND old keys still in source:")
        for fpath, key_preview in found_issues:
            print(f"      {fpath}: {key_preview}")
        return False
    else:
        print("  [OK] No old hardcoded keys found in source.")
        return True


def step_2_generate_env():
    """Generate fresh .env file from .env.example."""
    print_header("STEP 2: Generating .env File")

    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"

    if env_path.exists():
        print(f"  [INFO] .env already exists at {env_path}")
        print("  [ACTION] Update it with your new rotated keys.")
        return

    if example_path.exists():
        content = example_path.read_text()
        env_path.write_text(content)
        print(f"  [OK] Created .env from .env.example")
        print(f"  [ACTION] Edit {env_path} with your new keys.")
    else:
        # Generate from scratch
        lines = [
            "# Mini-Quant-Fund Environment Variables",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "# Alpaca API (Paper Trading)",
            "ALPACA_API_KEY=",
            "ALPACA_SECRET_KEY=",
            "ALPACA_BASE_URL=https://paper-api.alpaca.markets",
            "",
            "# Data Providers",
            "POLYGON_API_KEY=",
            "NEWS_API_KEY=",
            "",
            "# AI Provider",
            "DEEPSEEK_API_KEY=",
            "",
            "# Telegram Bot (optional)",
            "TELEGRAM_BOT_TOKEN=",
            "TELEGRAM_CHAT_ID=",
            "",
            "# Database (Production)",
            "DATABASE_URL=postgresql://quant:quant@localhost:5432/mini_quant_fund",
            "",
            "# Interactive Brokers (optional)",
            "IB_HOST=127.0.0.1",
            "IB_PORT=4002",
            "",
        ]
        env_path.write_text("\n".join(lines))
        print(f"  [OK] Created fresh .env at {env_path}")


def step_3_gitignore_check():
    """Ensure .gitignore covers all sensitive files."""
    print_header("STEP 3: Verifying .gitignore")

    gitignore_path = PROJECT_ROOT / ".gitignore"
    required_entries = [
        ".env",
        ".env.local",
        ".env.production",
        "*.pem",
        "*.key",
        ".secrets.baseline",
    ]

    if gitignore_path.exists():
        content = gitignore_path.read_text()
        missing = [e for e in required_entries if e not in content]
        if missing:
            with open(gitignore_path, "a") as f:
                f.write("\n\n# Security — auto-added\n")
                for entry in missing:
                    f.write(f"{entry}\n")
            print(f"  [OK] Added {len(missing)} entries to .gitignore")
        else:
            print("  [OK] .gitignore covers all sensitive patterns.")
    else:
        print("  [WARN] No .gitignore found! Creating one.")
        with open(gitignore_path, "w") as f:
            f.write("# Security\n")
            for entry in required_entries:
                f.write(f"{entry}\n")


def step_4_precommit_hook():
    """Install detect-secrets pre-commit hook."""
    print_header("STEP 4: Pre-commit Secret Detection")

    hook_dir = PROJECT_ROOT / ".git" / "hooks"
    hook_path = hook_dir / "pre-commit"

    if not hook_dir.exists():
        print("  [SKIP] No .git directory — not a git repo.")
        return

    hook_content = """#!/bin/sh
# Pre-commit hook: Scan for secrets before committing
# Auto-generated by rotate_keys.py

echo "Scanning for secrets..."

# Check for common secret patterns
if git diff --cached --diff-filter=ACM | grep -iE '(api[_-]?key|secret|password|token)\\s*[:=]\\s*["\\''][A-Za-z0-9]{16,}' 2>/dev/null; then
    echo ""
    echo "ERROR: Potential secret detected in staged files!"
    echo "Please remove secrets and use environment variables."
    exit 1
fi

# Run detect-secrets if available
if command -v detect-secrets &> /dev/null; then
    detect-secrets scan --baseline .secrets.baseline
fi

echo "Secret scan passed."
exit 0
"""

    hook_path.write_text(hook_content)
    # Make executable (Unix)
    try:
        os.chmod(hook_path, 0o755)
    except Exception:
        pass
    print("  [OK] Pre-commit hook installed.")


def step_5_print_rotation_guide():
    """Print detailed key rotation instructions."""
    print_header("STEP 5: Key Rotation Instructions")

    print("\n  You MUST rotate the following keys because")
    print("  they were previously exposed in source code:\n")

    for key_name, info in EXPOSED_KEYS.items():
        if not info["old_value"]:
            continue
        print(f"  --- {key_name} ({info['service']}) ---")
        print(f"  URL: {info['rotation_url']}")
        print(f"  Steps:")
        for line in info["instructions"].split("\n"):
            print(f"    {line}")
        print()


def step_6_validate_env():
    """Validate .env file has all required keys set."""
    print_header("STEP 6: Validating .env Configuration")

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print("  [FAIL] No .env file found. Run this script first.")
        return False

    content = env_path.read_text()
    required = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "POLYGON_API_KEY",
        "NEWS_API_KEY",
    ]

    all_set = True
    for key in required:
        line = [l for l in content.split("\n") if l.startswith(f"{key}=")]
        if not line or line[0].strip().endswith("="):
            print(f"  [!] {key} is NOT SET")
            all_set = False
        else:
            val = line[0].split("=", 1)[1].strip()
            if val.startswith("your_") or not val:
                print(f"  [!] {key} has placeholder value")
                all_set = False
            else:
                masked = val[:4] + "*" * (len(val) - 4)
                print(f"  [OK] {key} = {masked}")

    return all_set


def main():
    """Run full key rotation and security setup."""
    print("\n" + "=" * 60)
    print("  MINI-QUANT-FUND v1.0.0")
    print("  API Key Rotation & Security Setup")
    print("=" * 60)

    # Step 1: Scan for old secrets
    clean = step_1_scan_for_secrets()

    # Step 2: Generate .env
    step_2_generate_env()

    # Step 3: Gitignore
    step_3_gitignore_check()

    # Step 4: Pre-commit hook
    step_4_precommit_hook()

    # Step 5: Rotation guide
    step_5_print_rotation_guide()

    # Step 6: Validate
    step_6_validate_env()

    print_header("NEXT STEPS")
    print("  1. Rotate all keys using the URLs above")
    print("  2. Paste new keys into .env")
    print("  3. Re-run this script to validate")
    print("  4. Start paper trading with:")
    print("     python scripts/paper_trading_launcher.py")
    print()


if __name__ == "__main__":
    main()
