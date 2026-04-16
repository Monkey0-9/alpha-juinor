"""
risk/kill_switch.py

Ticket 24: Global Kill Conditions.
Enforces hard stops for the entire trading system.
"""

import logging
from enum import Enum
from typing import Any, Tuple

logger = logging.getLogger("KILL_SWITCH")


class KillSwitchReason(Enum):
    """Enumeration of kill switch activation reasons."""

    REGIME_CONFIDENCE_LOW = "Regime confidence below threshold"
    DATA_QUALITY_LOW = "Data quality below threshold"
    PORTFOLIO_RISK_HIGH = "Portfolio risk exceeds limit"
    AUDIT_STORAGE_ERROR = "Audit storage unwritable"
    SYSTEM_ERROR = "Unexpected system error"
    MANUAL_TRIGGER = "Manual kill switch triggered"
    OK = "System safe"


class GlobalKillSwitch:
    """
    Central safety monitor that enforces shutdown conditions.
    """

    def __init__(self):
        self.min_regime_confidence = 0.5
        self.min_data_quality = 0.8
        self.max_cvar_limit = -0.15  # 15% Daily CVaR Limit implies likely ruin

    def verify_safety(self, agent: Any) -> Tuple[bool, str]:
        """
        Verify all system invariants.

        Args:
            agent: The InstitutionalLiveAgent instance (or mock)

        Returns:
            (is_safe: bool, reason: str)
        """

        # 1. Regime Confidence Check
        # Agent has self.regime_controller.current_regime
        # We need confidence.
        # RegimeController.detect_regime returns Enum.
        # But dashboard holds 'confidence'.
        # Let's assume RegimeController handles confidence internally or we use Dashboard state.
        if hasattr(agent, "dashboard") and agent.dashboard:
            regime_state = agent.dashboard.state.get("regime", {})
            conf = regime_state.get("confidence", 1.0)
            if conf < self.min_regime_confidence:
                return (
                    False,
                    f"Regime Confidence {conf:.2f} < {self.min_regime_confidence}",
                )

        # 2. Data Health Check
        # Check metrics or dashboard
        if hasattr(agent, "dashboard") and agent.dashboard:
            data_health = agent.dashboard.state.get("data_health", {})
            quality = data_health.get("avg_quality", 1.0)
            if quality < self.min_data_quality:
                return False, f"Data Quality {quality:.2f} < {self.min_data_quality}"

        # 3. Portfolio Risk Check (CVaR)
        if hasattr(agent, "dashboard") and agent.dashboard:
            port_state = agent.dashboard.state.get("portfolio", {})
            current_cvar = port_state.get("cvar_95", 0.0)
            # CVaR is negative. -0.20 < -0.15 is True (More risk)
            if current_cvar < self.max_cvar_limit:
                return (
                    False,
                    f"Portfolio CVaR {current_cvar:.1%} exceeds limit {self.max_cvar_limit:.1%}",
                )

        # 4. Audit DB Health (Check if writable)
        # Check if audit directory exists and is writable
        try:
            with open("runtime/safety.check", "w") as f:
                f.write("ok")
        except Exception as e:
            return False, f"Audit Storage Unwritable: {e}"

        return True, "OK"


class DistributedKillSwitch(GlobalKillSwitch):
    """
    Distributed version of GlobalKillSwitch for multi-node trading systems.
    Extends GlobalKillSwitch with distributed coordination capabilities.
    """

    def __init__(self, node_id: str = "primary"):
        super().__init__()
        self.node_id = node_id
        self.is_primary = node_id == "primary"
        logger.info(
            f"[DistributedKillSwitch] Initialized node={node_id}, primary={self.is_primary}"
        )

    def verify_safety_distributed(
        self, agent: Any, other_nodes: list = None
    ) -> Tuple[bool, str]:
        """
        Verify safety with coordination across multiple nodes.

        Args:
            agent: The trading agent instance
            other_nodes: List of other node statuses

        Returns:
            (is_safe: bool, reason: str)
        """
        # First check local safety
        is_safe, reason = self.verify_safety(agent)

        if not is_safe:
            logger.warning(
                f"[DistributedKillSwitch({self.node_id})] Local safety check failed: {reason}"
            )
            return False, reason

        # If primary, consider other nodes
        if self.is_primary and other_nodes:
            for node_status in other_nodes:
                if not node_status.get("safe", False):
                    logger.warning(
                        f"[DistributedKillSwitch({self.node_id})] Remote node {node_status.get('node_id')} unsafe: "
                        f"{node_status.get('reason', 'Unknown')}"
                    )
                    return False, f"Remote node unsafe: {node_status.get('node_id')}"

        return True, "All nodes safe"
