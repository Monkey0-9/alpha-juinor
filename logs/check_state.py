"""Diagnostic: inspect running nexus python processes and scheduled task state."""

import subprocess
import os

root = r"c:\mini-quant-fund"

print("=" * 60)
print("PYTHON EXECUTABLES & VENV CHECK")
print("=" * 60)
venvs = [".venv", ".venv-1", "venv"]
for v in venvs:
    py = os.path.join(root, v, "Scripts", "python.exe")
    if os.path.exists(py):
        try:
            r = subprocess.run(
                [py, "-c", "import sys;print(sys.version.split()[0])"],
                capture_output=True,
                text=True,
            )
            print(f"{v}: exists, python {r.stdout.strip()}")
        except Exception as e:
            print(f"{v}: exists, error {e}")
    else:
        print(f"{v}: MISSING")

print()
print("=" * 60)
print("GETTING CMDLINES OF RUNNING PROCESSES")
print("=" * 60)
try:
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "ForEach-Object { '{0}|{1}'.f('$($_.ProcessId)',$($_.CommandLine)) }",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(out.stdout)
    if out.stderr:
        print("STDERR:", out.stderr[:2000])
except Exception as e:
    print("Error enumerating processes:", e)

print()
print("=" * 60)
print("SCHEDULED TASK CHECK")
print("=" * 60)
try:
    out = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-ScheduledTask -TaskName 'Nexus24x7TradingPlatform' -ErrorAction SilentlyContinue | "
            "ForEach-Object { 'State={0},Path={1}'.f($_.State,$_.TaskPath) }",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    print("Task:", out.stdout.strip() or "(not found or no access)")
    if out.stderr:
        print("STDERR:", out.stderr[:1000])
except Exception as e:
    print("Error checking task:", e)

print("DONE")
