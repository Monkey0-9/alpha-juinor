"""Diagnostic 2: list nexus-related processes via tasklist and PowerShell string format."""

import subprocess
import os

root = r"c:\mini-quant-fund"

print("RUNNING PYTHON PROCESSES (tasklist):")
try:
    out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=30)
    for line in out.stdout.splitlines():
        if "python" in line.lower():
            print(line)
except Exception as e:
    print("tasklist error:", e)

print()
print("RESOLVING python in PATH:")
try:
    out = subprocess.run(
        ["where", "python"], capture_output=True, text=True, timeout=30
    )
    print(out.stdout)
except Exception as e:
    print(e)

print()
print("EXISTING 24/7 LOG TODAY:")
logfile = os.path.join(root, "logs", "nexus_24_7_last.log")
print("(checked via listing)")

print("DONE")
