#!/usr/bin/env python3
"""
Windows Service Manager for Nexus 24/7 Trading Platform
Installs, starts, stops, and manages Nexus as a Windows Service.

Usage:
  python nexus_service_manager.py install    # Install service
  python nexus_service_manager.py start      # Start service
  python nexus_service_manager.py stop       # Stop service
  python nexus_service_manager.py remove     # Uninstall service
  python nexus_service_manager.py status     # Check service status
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WindowsServiceManager:
    """Manages Nexus as a Windows Service."""

    SERVICE_NAME = "NexusTradingPlatform24x7"
    SERVICE_DISPLAY_NAME = "Nexus 24/7 Trading Platform"
    SERVICE_DESCRIPTION = "Continuous quantitative trading platform with auto-restart"

    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.python_exe = sys.executable
        self.script_path = self.project_dir / "nexus_24_7.py"

    def _run_command(self, cmd: list, check: bool = True) -> bool:
        """Run a command and return success status."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr and result.returncode != 0:
                logger.error(result.stderr)
            return result.returncode == 0
        except Exception as exc:
            logger.error(f"Error running command: {exc}")
            return False

    def install(self):
        """Install Nexus as Windows Service."""
        logger.info(f"Installing {self.SERVICE_DISPLAY_NAME}...")

        # Check if service already exists
        status = self._run_command(
            ["sc", "query", self.SERVICE_NAME],
            check=False
        )

        if status:
            logger.warning(f"Service {self.SERVICE_NAME} already exists.")
            response = input("Remove existing service first? (y/n): ")
            if response.lower() == 'y':
                self.remove()
            else:
                return False

        # Create service
        service_cmd = (
            f'"{self.python_exe}" "{self.script_path}"'
        )

        cmd = [
            "sc",
            "create",
            self.SERVICE_NAME,
            f"binPath={service_cmd}",
            f"DisplayName={self.SERVICE_DISPLAY_NAME}",
            "start=auto",
            "type=own"
        ]

        if self._run_command(cmd):
            logger.info(f"✓ Service {self.SERVICE_NAME} installed successfully")

            # Set description
            desc_cmd = [
                "sc",
                "description",
                self.SERVICE_NAME,
                self.SERVICE_DESCRIPTION
            ]
            self._run_command(desc_cmd)

            logger.info(f"Start the service with: net start {self.SERVICE_NAME}")
            return True
        else:
            logger.error("Failed to install service")
            return False

    def start(self):
        """Start the Windows Service."""
        logger.info(f"Starting {self.SERVICE_NAME}...")
        if self._run_command(["net", "start", self.SERVICE_NAME]):
            logger.info(f"✓ Service {self.SERVICE_NAME} started")
            return True
        else:
            logger.error(f"Failed to start service {self.SERVICE_NAME}")
            return False

    def stop(self):
        """Stop the Windows Service."""
        logger.info(f"Stopping {self.SERVICE_NAME}...")
        if self._run_command(["net", "stop", self.SERVICE_NAME], check=False):
            logger.info(f"✓ Service {self.SERVICE_NAME} stopped")
            return True
        else:
            logger.warning(f"Service {self.SERVICE_NAME} may not have been running")
            return True

    def remove(self):
        """Uninstall the Windows Service."""
        logger.warning(f"Removing {self.SERVICE_NAME}...")

        # Stop service first
        self.stop()

        # Remove service
        if self._run_command(["sc", "delete", self.SERVICE_NAME], check=False):
            logger.info(f"✓ Service {self.SERVICE_NAME} removed")
            return True
        else:
            logger.error(f"Failed to remove service {self.SERVICE_NAME}")
            return False

    def status(self):
        """Check service status."""
        logger.info(f"Checking status of {self.SERVICE_NAME}...")
        cmd = ["sc", "query", self.SERVICE_NAME]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Service {self.SERVICE_NAME} not found or error occurred")
            logger.error(result.stderr)
            return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Check for admin privileges
    try:
        import ctypes
        is_admin = ctypes.windll.shell.IsUserAnAdmin()
    except Exception:
        is_admin = False

    if not is_admin:
        print("ERROR: This script requires administrator privileges!")
        print("Please run as Administrator (right-click cmd.exe and select 'Run as administrator')")
        sys.exit(1)

    manager = WindowsServiceManager()
    command = sys.argv[1].lower()

    if command == "install":
        success = manager.install()
    elif command == "start":
        success = manager.start()
    elif command == "stop":
        success = manager.stop()
    elif command == "remove":
        success = manager.remove()
    elif command == "status":
        success = manager.status()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
