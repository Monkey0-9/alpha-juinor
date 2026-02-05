import sys
import os
import time
import yaml

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk.kill_switch import DistributedKillSwitch, KillSwitchState, KillSwitchReason

def test_kill_switch_fallback():
    print("Testing Kill Switch Fallback (REDIS-LESS)...")

    # Load config
    with open("configs/kill_switch_config.yaml", "r") as f:
        config = yaml.safe_load(f)['kill_switch']

    # Force Redis-less mode by using a dummy host
    config['redis_host'] = "invalid_host_12345"
    config['redis_port'] = 9999

    # Initialize
    print("Initializing KillSwitch with invalid Redis...")
    ks = DistributedKillSwitch(config)

    print(f"Current Redis Mode: {ks.redis_mode}")
    print(f"Current State: {ks.state}")

    assert ks.redis_mode == False, "Redis mode should be False for invalid host"
    assert ks.state == KillSwitchState.ARMED, "Initial state should be ARMED"

    # Check if local state file exists
    if os.path.exists(ks.local_fallback_file):
        print(f"Local state file found: {ks.local_fallback_file}")
    else:
        print("ERROR: Local state file NOT created!")
        sys.exit(1)

    # Test Trigger
    print("\nTriggering Manual Kill...")
    ks.trigger(KillSwitchReason.MANUAL, "Manual test trigger", "tester")

    print(f"New State: {ks.state}")
    assert ks.state == KillSwitchState.TRIGGERED, "State should be TRIGGERED after trigger"

    # Check if local state file updated
    with open(ks.local_fallback_file, 'r') as f:
        local_content = f.read().strip()
    print(f"Local file content: {local_content}")
    assert local_content == "triggered", "Local file should say triggered"

    # Verify check_and_block does not crash
    print("Verifying check_and_block...")
    should_block = ks.check_and_block()
    print(f"Should block: {should_block}")
    assert should_block == True, "Should block when triggered"

    print("\nSUCCESS: Kill Switch local fallback is operational.")

if __name__ == "__main__":
    test_kill_switch_fallback()
