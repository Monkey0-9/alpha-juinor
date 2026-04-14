#!/usr/bin/env python3
"""
Complete test runner - verifies all tests pass.
"""
import subprocess
import sys
from pathlib import Path

def main() -> int:
    test_file = Path(__file__).resolve()
    repo_root = test_file.parents[1]
    test_dir = test_file.parent

    print("=" * 80)
    print("RUNNING COMPLETE TEST SUITE")
    print("=" * 80)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_dir),
            "--import-mode=importlib",
            "-v",
            "--tb=short",
        ],
        cwd=repo_root,
        check=False,
    )

    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("PASS: ALL TESTS PASSED!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAIL: SOME TESTS FAILED")
        print("=" * 80)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
