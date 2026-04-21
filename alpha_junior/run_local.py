#!/usr/bin/env python3
"""
Alpha Junior - FULLY LOCAL LAUNCHER
No Docker, no PostgreSQL, no Redis needed!
Just Python 3.11+ and run this file.
"""

import subprocess
import sys
import os

def main():
    print("="*60)
    print("  Alpha Junior - LOCAL MODE (No Docker!)")
    print("="*60)
    print()
    
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    
    # Step 1: Check Python version
    print("[1/4] Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11+ required!")
        print(f"   You have: Python {version.major}.{version.minor}")
        print("   Download from: https://python.org/downloads")
        return 1
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    
    # Step 2: Install requirements
    print("\n[2/4] Installing Python packages...")
    req_file = os.path.join(backend_dir, "requirements_local.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        cwd=backend_dir,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Failed to install packages")
        print(result.stderr)
        return 1
    print("✓ Packages installed")
    
    # Step 3: Create database
    print("\n[3/4] Setting up database...")
    from backend.app.db.session_local import init_db, engine
    import asyncio
    asyncio.run(init_db())
    print("✓ SQLite database ready")
    
    # Step 4: Start server
    print("\n[4/4] Starting server...")
    print()
    print("="*60)
    print("  🚀 Alpha Junior is RUNNING!")
    print("="*60)
    print()
    print("  📍 API:       http://localhost:8000")
    print("  📚 Docs:      http://localhost:8000/api/v1/docs")
    print("  🔥 Health:    http://localhost:8000/health")
    print()
    print("  Press Ctrl+C to stop")
    print("="*60)
    print()
    
    # Start uvicorn
    os.chdir(backend_dir)
    sys.path.insert(0, backend_dir)
    
    import uvicorn
    uvicorn.run(
        "app.main_local:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
