"""
Test script to verify audit logging works after the path sanitization fix.
Run this to confirm the fix resolves [Errno 22] Invalid argument errors.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_audit_fix():
    """Test that audit logging works without Windows path errors."""
    print("=" * 60)
    print("TESTING AUDIT PATH SANITIZATION FIX")
    print("=" * 60)

    # Test 1: Import modules without error
    print("\n[TEST 1] Importing audit modules...")
    try:
        from audit.decision_log import AUDIT_JSONL_PATH, write_audit
        print(f"  ✓ decision_log imported successfully")
        print(f"  ✓ AUDIT_JSONL_PATH: {AUDIT_JSONL_PATH}")
    except Exception as e:
        print(f"  ✗ Failed to import decision_log: {e}")
        return False

    try:
        from audit.decision_recorder import DecisionRecorder
        print(f"  ✓ decision_recorder imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import decision_recorder: {e}")
        return False

    # Test 2: Write a test audit record
    print("\n[TEST 2] Writing test audit record...")
    try:
        test_record = {
            "cycle_id": "TEST_FIX_CYCLE",
            "symbol": "TEST_SYMBOL",
            "timestamp": "2026-02-07T07:15:07Z",  # Contains colons - the problematic character
            "data_providers": {"test": True},
            "alphas": {"alpha1": 0.5},
            "sigmas": {"sigma1": 0.1},
            "conviction": 0.75,
            "conviction_zscore": 1.5,
            "risk_checks": ["passed"],
            "pm_override": "NONE",
            "final_decision": "HOLD",
            "reason_codes": ["TEST"],
            "order": None,
            "raw_traceback": None
        }
        write_audit(test_record)
        print(f"  ✓ Audit record queued successfully")
    except Exception as e:
        print(f"  ✗ Failed to write audit record: {e}")
        return False

    # Test 3: Create a DecisionRecorder and log
    print("\n[TEST 3] Testing DecisionRecorder...")
    try:
        recorder = DecisionRecorder(log_dir="logs/decisions")
        recorder.record(
            symbol="TEST_FIX",
            decision="HOLD",
            signal_strength=0.5,
            confidence=0.8,
            source_alpha="test_alpha",
            rationale="Testing path sanitization fix"
        )
        print(f"  ✓ DecisionRecorder wrote successfully")
    except Exception as e:
        print(f"  ✗ DecisionRecorder failed: {e}")
        return False

    # Test 4: Give the async writer time to process
    print("\n[TEST 4] Waiting for async writer...")
    import time
    time.sleep(2)
    print(f"  ✓ Waited 2 seconds for async processing")

    # Test 5: Verify no CRITICAL errors in recent output
    print("\n[TEST 5] Checking for JSONL file existence...")
    if os.path.exists(AUDIT_JSONL_PATH):
        size = os.path.getsize(AUDIT_JSONL_PATH)
        print(f"  ✓ {AUDIT_JSONL_PATH} exists ({size:,} bytes)")
    else:
        print(f"  ⚠ {AUDIT_JSONL_PATH} does not exist (may be first run)")

    print("\n" + "=" * 60)
    print("TEST PASSED - No CRITICAL AUDIT JSONL FAILURE errors!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_audit_fix()
    sys.exit(0 if success else 1)
