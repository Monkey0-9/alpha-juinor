#!/usr/bin/env python
"""
Test runner with proper sys.path setup.
Run this to execute all tests correctly.
"""

import os
import subprocess
import sys

# Ensure current directory is in path
sys.path.insert(0, os.getcwd())

# Run pytest with correct configuration
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--import-mode=importlib",
        "--tb=short",
        "-v",
        "--color=yes",
    ],
    cwd=os.getcwd(),
)

sys.exit(result.returncode)
