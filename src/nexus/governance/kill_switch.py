"""
governance/kill_switch.py

Automated Kill Switch & SLA Monitor.
Continuously checks system health against SLAs.
Triggers HALT if critical thresholds breached.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger("KILL_SWITCH")

# SLA Thresholds
SLA_MIN_DATA_QUALITY = 0.60
SLA_MAX_PROVIDER_FAIL_RATE = 0.10
SLA_AUDIT_HEALTH = True

class AutoKillSwitch:
    def __init__(self, db_manager):
        self.db = db_manager
        self.kill_file = "runtime/KILL_SWITCH"

    def check_slas(self, stats: Dict[str, Any]) -> bool:
        """
        Run all SLA checks.
        Returns True if system is HEALTHY.
        Returns False (and triggers kill) if CRITICAL FAILURE.
        """
        breaches = []

        # 1. Data Quality SLA
        avg_quality = stats.get("avg_data_quality", 1.0)
        if avg_quality < SLA_MIN_DATA_QUALITY:
            breaches.append(f"DATA_QUALITY ({avg_quality:.2f} < {SLA_MIN_DATA_QUALITY})")

        # 2. Provider Failure Rate
        total = stats.get("total_symbols", 0)
        failed = stats.get("failed", 0)
        if total > 0:
            fail_rate = failed / total
            if fail_rate > SLA_MAX_PROVIDER_FAIL_RATE:
                breaches.append(f"PROVIDER_FAILURE ({fail_rate:.2%} > {SLA_MAX_PROVIDER_FAIL_RATE:%})")

        # 3. Audit Health
        if not self.db.check_table_exists("decisions"):
             breaches.append("AUDIT_DB_MISSING")

        if breaches:
            self._trigger_halt(breaches)
            return False

        return True

    def _trigger_halt(self, reasons: list):
        """Execute Kill Switch."""
        logger.critical("=" * 60)
        logger.critical("!!! AUTOMATED KILL SWITCH TRIGGERED !!!")
        logger.critical(f"Reasons: {', '.join(reasons)}")
        logger.critical("System HALTED to prevent damage.")
        logger.critical("=" * 60)

        with open(self.kill_file, "w") as f:
            f.write(f"AUTOMATED_HALT\nTimestamp: {os.path.abspath(self.kill_file)}\nReasons: {reasons}")

        # In production this would also:
        # - Cancel all open orders
        # - Send PagerDuty alert
        # - Flush logs
