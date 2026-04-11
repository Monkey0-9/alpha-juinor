#!/usr/bin/env python3
"""
Complete test runner - verifies all tests pass.
"""
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run all tests with proper setup
print("=" * 80)
print("RUNNING COMPLETE TEST SUITE")
print("=" * 80)

result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--import-mode=importlib",
        "-v",
        "--tb=short",
    ]
)

if result.returncode == 0:
    print("\n" + "=" * 80)
    print("PASS: ALL TESTS PASSED!")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("FAIL: SOME TESTS FAILED")
    print("=" * 80)

sys.exit(result.returncode)
