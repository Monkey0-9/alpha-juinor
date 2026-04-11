"""
GitHub Secrets Setup Helper
=============================
Automates the setup of GitHub repository secrets
for the CI/CD pipeline.

Usage:
    python scripts/setup_github_secrets.py

Requires:
    - GitHub CLI (gh) installed: https://cli.github.com
    - Authenticated: gh auth login
    OR
    - Manual setup via GitHub web UI
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Secrets required for the CI/CD pipeline
REQUIRED_SECRETS = {
    "ALPACA_API_KEY": {
        "description": "Alpaca Paper Trading API Key",
        "used_in": "Unit/integration tests",
        "critical": True,
    },
    "ALPACA_SECRET_KEY": {
        "description": "Alpaca Paper Trading Secret Key",
        "used_in": "Unit/integration tests",
        "critical": True,
    },
    "POLYGON_API_KEY": {
        "description": "Polygon.io API Key",
        "used_in": "Data provider tests",
        "critical": False,
    },
    "NEWS_API_KEY": {
        "description": "NewsAPI.org API Key",
        "used_in": "Sentiment analysis tests",
        "critical": False,
    },
    "DEEPSEEK_API_KEY": {
        "description": "DeepSeek AI API Key",
        "used_in": "ML/AI strategy tests",
        "critical": False,
    },
    "TELEGRAM_BOT_TOKEN": {
        "description": "Telegram Bot Token",
        "used_in": "Alert notifications",
        "critical": False,
    },
    "TELEGRAM_CHAT_ID": {
        "description": "Telegram Chat ID",
        "used_in": "Alert notifications",
        "critical": False,
    },
}


def check_gh_cli() -> bool:
    """Check if GitHub CLI is installed."""
    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_gh_auth() -> bool:
    """Check if user is authenticated with gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_repo_name() -> str:
    """Get GitHub repository name from git remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        url = result.stdout.strip()
        # Parse owner/repo from URL
        if "github.com" in url:
            parts = url.rstrip(".git").split("/")
            return f"{parts[-2]}/{parts[-1]}"
    except Exception:
        pass
    return ""


def set_secret_gh(name: str, value: str) -> bool:
    """Set a GitHub secret using gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "secret", "set", name, "--body", value],
            capture_output=True, text=True,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode == 0
    except Exception:
        return False


def setup_via_cli():
    """Set up secrets using GitHub CLI."""
    print("\n  Setting up secrets via GitHub CLI...")

    # Load .env
    env_path = PROJECT_ROOT / ".env"
    env_vars = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()

    for name, info in REQUIRED_SECRETS.items():
        value = env_vars.get(name, "")
        if not value or value.startswith("your_"):
            if info["critical"]:
                print(f"  [!] {name}: NOT SET (CRITICAL)")
            else:
                print(f"  [-] {name}: skipped (optional)")
            continue

        success = set_secret_gh(name, value)
        if success:
            print(f"  [OK] {name}: set successfully")
        else:
            print(f"  [!] {name}: failed to set")


def print_manual_guide():
    """Print manual GitHub Secrets setup guide."""
    repo = get_repo_name() or "YOUR_ORG/mini-quant-fund"

    print(f"""
  =====================================================
  MANUAL GITHUB SECRETS SETUP
  =====================================================

  Repository: {repo}

  1. Go to: https://github.com/{repo}/settings/secrets/actions

  2. Click "New repository secret" for each of the following:

""")
    for name, info in REQUIRED_SECRETS.items():
        critical = " (CRITICAL)" if info["critical"] else " (optional)"
        print(f"     Name:  {name}")
        print(f"     Value: <paste from your .env file>")
        print(f"     Info:  {info['description']}{critical}")
        print(f"     Used:  {info['used_in']}")
        print()

    print("""
  3. Also create these ENVIRONMENT secrets:

     Go to: Settings > Environments > New environment

     Create "staging" environment:
       - No approval required
       - Deploy from main branch only

     Create "production" environment:
       - Require approval (add yourself as reviewer)
       - Deploy from release/* branches only
       - Add a 15-minute wait timer

  4. Verify by running a push to trigger the pipeline:
     git push origin main
""")


def main():
    print("\n" + "=" * 60)
    print("  MINI-QUANT-FUND v1.0.0")
    print("  GitHub Secrets Setup")
    print("=" * 60)

    # Check gh CLI
    if check_gh_cli() and check_gh_auth():
        print("\n  GitHub CLI detected and authenticated.")
        response = input(
            "  Auto-configure secrets from .env? [y/N]: "
        )
        if response.lower() == "y":
            setup_via_cli()
            return

    # Manual fallback
    print_manual_guide()


if __name__ == "__main__":
    main()
