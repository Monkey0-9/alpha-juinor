#!/usr/bin/env python3
"""
Script to manually trigger the Distributed Kill Switch.
Usage: python scripts/trigger_kill_switch.py --reason "Manual Override" --timeout 3600
"""

import sys
import os
import argparse
import yaml
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.kill_switch import DistributedKillSwitch, KillSwitchReason

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ManualKillSwitch")

def main():
    parser = argparse.ArgumentParser(description="Manually trigger the Distributed Kill Switch")
    parser.add_argument("--reason", type=str, default="Manual Emergency Override", help="Reason for triggering")
    parser.add_argument("--component", type=str, default="Operator", help="Component triggering the switch")
    args = parser.parse_args()

    try:
        # Load config
        config_path = "configs/kill_switch_config.yaml"
        if not os.path.exists(config_path):
             logger.error(f"Config file not found at {config_path}")
             sys.exit(1)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Initialize Kill Switch
        ks = DistributedKillSwitch(config['kill_switch'])

        logger.info(f"Attempting to trigger Kill Switch. Reason: {args.reason}")

        success = ks.trigger(
            reason=KillSwitchReason.MANUAL_OVERRIDE,
            description=args.reason,
            source_component=args.component
        )

        if success:
            logger.info("SUCCESS: Kill Switch Triggered. System should halt.")
        else:
            logger.error("FAILED: Could not trigger Kill Switch. Check logs/Redis.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
