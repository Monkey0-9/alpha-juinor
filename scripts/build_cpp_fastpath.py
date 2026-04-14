#!/usr/bin/env python3
"""
Build the C++ fast decision core shared library.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "cpp" / "fast_decision"
    build_dir = source_dir / "build"

    if shutil.which("cmake") is None:
        print("cmake not found on PATH. Install CMake first.")
        return 1

    build_dir.mkdir(parents=True, exist_ok=True)

    configure_cmd = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]

    print("Configuring C++ fastpath...")
    subprocess.run(configure_cmd, check=True)
    print("Building C++ fastpath...")
    subprocess.run(build_cmd, check=True)

    artifacts = list(build_dir.rglob("fast_decision_core.dll"))
    artifacts += list(build_dir.rglob("libfast_decision_core.so"))
    artifacts += list(build_dir.rglob("libfast_decision_core.dylib"))

    if not artifacts:
        print("Build finished, but no shared library artifact found.")
        return 2

    lib_path = artifacts[0].resolve()
    print(f"Built: {lib_path}")
    print(f"Set env var for runtime: MQF_CPP_CORE_LIB={lib_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
