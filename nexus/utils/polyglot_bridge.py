import json
import os
import platform
import shutil
import subprocess
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PolyglotBridge:
    """Cross-platform bridge for high-performance satellite engines."""

    @staticmethod
    def calculate_risk_rust(
        returns: List[float],
        conf: float = 0.95,
    ) -> Dict[str, Any]:
        """Execute Rust Risk Engine for VaR calculation."""
        input_data = json.dumps(
            {"returns": returns, "confidence_level": conf}
        )
        try:
            if not shutil.which("cargo"):
                return {
                    "var": 0.0,
                    "expected_shortfall": 0.0,
                    "status": "RUST_NOT_INSTALLED",
                }

            cmd = [
                "cargo", "run", "--quiet",
                "--manifest-path",
                "nexus/polyglot/rust_risk_engine/Cargo.toml",
                "--", input_data,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except Exception as exc:
            logger.debug(f"Rust Risk Engine fallback: {exc}")
            return {
                "var": 0.0,
                "expected_shortfall": 0.0,
                "status": "FALLBACK",
            }

    @staticmethod
    def audit_platform_go(
        backend_url: str = "http://127.0.0.1:8000",
    ) -> Dict[str, Any]:
        """Execute Go Platform Auditor for health checks."""
        try:
            # Platform-aware binary selection
            if platform.system() == "Windows":
                exe_path = "nexus/polyglot/go_auditor/auditor.exe"
            else:
                exe_path = "nexus/polyglot/go_auditor/auditor"

            if not os.path.exists(exe_path):
                return {
                    "overall_health": "DEGRADED",
                    "results": [
                        {
                            "service": "auditor",
                            "status": "BINARY_MISSING",
                        }
                    ],
                }

            cmd = [exe_path, backend_url]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except Exception as exc:
            logger.error(f"Go Auditor failure: {exc}")
            return {"overall_health": "UNKNOWN", "results": []}

    @staticmethod
    def validate_order_zig(
        order: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Zig Validator for order verification."""
        try:
            if not shutil.which("zig"):
                return {
                    "valid": True,
                    "error": "ZIG_NOT_INSTALLED",
                }

            input_data = json.dumps(order)
            cmd = [
                "zig", "run",
                "nexus/polyglot/zig_validator/main.zig",
                "--", input_data,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except Exception as exc:
            logger.debug(f"Zig Validator fallback: {exc}")
            return {
                "valid": True,
                "error": "VALIDATOR_OFFLINE",
            }
