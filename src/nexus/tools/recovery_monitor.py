"""
Recovery Operations Monitor
==========================
Automated monitoring for all 3 recovery stages.
Run this periodically to check system health and stage completion.
"""
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
LOG_DIR = Path("logs")
SYSTEM_LOG = Path("system.log")


def check_stage_1():
    """
    Stage 1: Contain Losses (Now - 48h)
    - Monitor first exit signals in logs
    - Verify no new buys executed
    """
    print("\n" + "=" * 60)
    print("STAGE 1: CONTAIN LOSSES")
    print("=" * 60)

    exit_signals = 0
    buy_attempts = 0
    buy_blocked = 0
    symmetry_logs = []

    # Check system.log
    if SYSTEM_LOG.exists():
        with open(SYSTEM_LOG, 'r', errors='ignore') as f:
            content = f.read()

        # Count exit-related signals
        exit_signals = len(re.findall(r'\[SELL\]|\[EXIT\]|action.*SELL', content, re.I))

        # Count buy attempts and blocks
        buy_attempts = len(re.findall(r'action.*BUY|side.*buy', content, re.I))
        buy_blocked = len(re.findall(r'\[EMERGENCY\].*BUY.*STOPPED', content, re.I))

        # Get symmetry logs
        symmetry_matches = re.findall(r'\[SYMMETRY\].*', content)
        symmetry_logs = symmetry_matches[-5:] if symmetry_matches else []

    # Status
    print(f"\n✅ Exit Signals Detected: {exit_signals}")
    print(f"🛑 Buy Attempts: {buy_attempts}")
    print(f"✅ Buys Blocked by Emergency Stop: {buy_blocked}")

    if symmetry_logs:
        print("\nRecent Symmetry Logs:")
        for log in symmetry_logs:
            print(f"  {log}")
    else:
        print("\n⚠️  No [SYMMETRY] logs found yet (hourly)")

    # Verification
    stage1_complete = exit_signals > 0 and (buy_attempts == 0 or buy_blocked >= buy_attempts)

    if stage1_complete:
        print("\n✅ STAGE 1 VERIFIED: System is generating exits and blocking buys")
    else:
        if exit_signals == 0:
            print("\n⚠️  Waiting for first exit signal...")
        if buy_attempts > buy_blocked:
            print("\n🚨 WARNING: Some buys may have slipped through!")

    return stage1_complete


def check_stage_2():
    """
    Stage 2: Stabilize (Week 1)
    - Reduce exposure to < 60%
    - Execute intelligent exits on worst performers
    """
    print("\n" + "=" * 60)
    print("STAGE 2: STABILIZE")
    print("=" * 60)

    # Import portfolio assessment
    try:
        import yfinance as yf

        positions = [
            {"symbol": "SIRI", "qty": 9057.8359, "cost": 20.60},
            {"symbol": "BKNG", "qty": 296.8295, "cost": 5118.60},
            {"symbol": "AAPL", "qty": 6.0, "cost": 257.86}
        ]

        total_invested = sum(p["qty"] * p["cost"] for p in positions)
        total_nav = 1706000  # Original NAV estimate

        exposure_pct = (total_invested / total_nav) * 100

        print(f"\nTotal Invested: ${total_invested:,.2f}")
        print(f"Estimated NAV: ${total_nav:,.2f}")
        print(f"Exposure: {exposure_pct:.1f}%")

        target = 60.0
        if exposure_pct <= target:
            print(f"\n✅ STAGE 2 COMPLETE: Exposure {exposure_pct:.1f}% <= {target}%")
            return True
        else:
            reduction_needed = total_invested - (total_nav * (target / 100))
            print(f"\n⚠️  Need to reduce ${reduction_needed:,.2f} to reach {target}% exposure")
            return False

    except Exception as e:
        print(f"\n⚠️  Could not calculate exposure: {e}")
        return False


def check_stage_3():
    """
    Stage 3: Rebuild (Week 2-4)
    - Achieve < 50% exposure
    - Remove Emergency Stop gradually
    - Re-enable selective buying (A+ grades only)
    """
    print("\n" + "=" * 60)
    print("STAGE 3: REBUILD (Preparation)")
    print("=" * 60)

    # Check if Emergency Stop is still active in main.py
    main_py = Path("main.py")
    emergency_stop_active = False

    if main_py.exists():
        with open(main_py, 'r', errors='ignore') as f:
            content = f.read()
        emergency_stop_active = "EMERGENCY" in content and "BUY STOPPED" in content

    print(f"\nEmergency Buy Stop in Code: {'ACTIVE' if emergency_stop_active else 'REMOVED'}")

    print("\n📋 Stage 3 Readiness Checklist:")
    print("  [ ] Exposure < 50%")
    print("  [ ] All tests passing")
    print("  [ ] Remove Emergency Stop (lines 1280-1286 in main.py)")
    print("  [ ] Monitor first A+ buys carefully")

    print("\n⏳ Stage 3 is a future phase - system will be ready when Stage 2 completes.")
    return False


def run_full_check():
    """Run all stage checks."""
    print("\n" + "=" * 60)
    print("VAST INTELLIGENCE: RECOVERY OPERATIONS MONITOR")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    s1 = check_stage_1()
    s2 = check_stage_2()
    s3 = check_stage_3()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Stage 1 (Contain):   {'✅ COMPLETE' if s1 else '🔄 IN PROGRESS'}")
    print(f"Stage 2 (Stabilize): {'✅ COMPLETE' if s2 else '🔄 PENDING'}")
    print(f"Stage 3 (Rebuild):   {'✅ READY' if s3 else '⏳ FUTURE'}")

    return s1, s2, s3


if __name__ == "__main__":
    run_full_check()
