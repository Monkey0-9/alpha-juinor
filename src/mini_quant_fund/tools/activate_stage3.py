"""
Stage 3 Activation Script
=========================
Run this ONLY when:
1. Stage 2 is complete (exposure < 50%)
2. All tests pass
3. You're ready to re-enable selective A+ buying

This script removes the Emergency Buy Stop and enables A+ only buying.
"""
import re
import shutil
from datetime import datetime
from pathlib import Path


def check_prerequisites():
    """Verify system is ready for Stage 3."""
    print("Checking Stage 3 prerequisites...")

    checks = {
        "tests_pass": False,
        "exposure_ok": False,
        "user_confirmed": False
    }

    # Would run tests here in production
    print("  [ ] All tests passing (run manually)")
    print("  [ ] Exposure < 50% (check with assess_portfolio.py)")
    print("  [ ] User confirmation required")

    return all(checks.values())


def backup_main():
    """Create backup of main.py before modification."""
    main_path = Path("main.py")
    backup_path = Path(f"main.py.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if main_path.exists():
        shutil.copy(main_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return True
    return False


def remove_emergency_stop():
    """
    Remove the Emergency Buy Stop from main.py.
    Lines 1280-1286 contain:

    # EMERGENCY BUY STOP (Phase 32)
    if target_weights[sym] > 0:
        logger.warning(...)
        target_weights[sym] = 0.0
        continue
    """
    main_path = Path("main.py")

    if not main_path.exists():
        print("❌ main.py not found!")
        return False

    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and comment out the emergency stop block
    pattern = r'(# EMERGENCY BUY STOP.*?continue\n)'

    def replacer(match):
        block = match.group(1)
        # Comment out the block instead of removing
        commented = '\n'.join(
            '# [STAGE 3 DISABLED] ' + line
            for line in block.split('\n')
        )
        return commented

    new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    if new_content != content:
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("✅ Emergency Buy Stop DISABLED (commented out)")
        return True
    else:
        print("⚠️  Could not find Emergency Stop block to disable")
        return False


def show_instructions():
    """Show manual instructions for Stage 3 activation."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          STAGE 3 ACTIVATION - MANUAL INSTRUCTIONS                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Before running this script:                                     ║
║                                                                  ║
║  1. Run: python -m unittest tests.test_symmetric_intelligence    ║
║          tests.test_vast_intelligence -v                         ║
║     ➜ ALL tests must pass                                        ║
║                                                                  ║
║  2. Run: python tools/assess_portfolio.py                        ║
║     ➜ Exposure must be < 50%                                     ║
║                                                                  ║
║  3. Review recent logs for any anomalies                         ║
║                                                                  ║
║  To activate Stage 3:                                            ║
║  ───────────────────                                             ║
║  Option A: Run this script with --activate flag                  ║
║            python tools/activate_stage3.py --activate            ║
║                                                                  ║
║  Option B: Manually comment out lines 1280-1286 in main.py       ║
║            (The EMERGENCY BUY STOP block)                        ║
║                                                                  ║
║  After activation:                                               ║
║  ─────────────────                                               ║
║  • Only A+ grade signals will execute buys                       ║
║  • Monitor first 2-3 buys VERY carefully                         ║
║  • Check [BRAIN] APPROVED logs for new positions                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 60)
    print("VAST INTELLIGENCE: STAGE 3 ACTIVATION")
    print("=" * 60)

    if "--activate" in sys.argv:
        print("\n⚠️  ACTIVATION MODE")
        confirm = input("Type 'ACTIVATE' to proceed: ")

        if confirm == "ACTIVATE":
            if backup_main():
                if remove_emergency_stop():
                    print("\n✅ STAGE 3 ACTIVATED!")
                    print("   Emergency Buy Stop has been disabled.")
                    print("   System will now accept A+ grade buys.")
                else:
                    print("\n❌ Activation failed. Check main.py manually.")
            else:
                print("\n❌ Backup failed. Aborting.")
        else:
            print("\n❌ Activation cancelled.")
    else:
        show_instructions()
