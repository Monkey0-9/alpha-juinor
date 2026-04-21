#!/usr/bin/env python3
"""
Alpha Junior - Complete System Test
Verifies all components are working
"""

import sys
import os
import sqlite3
import requests
import time

print("="*60)
print("  Alpha Junior - System Verification")
print("="*60)
print()

errors = []

# Test 1: Check Python version
print("[1/7] Checking Python...")
version = sys.version_info
if version.major >= 3 and version.minor >= 11:
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"   ⚠ Python {version.major}.{version.minor} (3.11+ recommended)")

# Test 2: Check dependencies
print("[2/7] Checking dependencies...")
try:
    import flask
    import flask_cors
    print("   ✓ Flask installed")
except ImportError as e:
    print(f"   ✗ Missing: {e}")
    print("   Run: pip install flask flask-cors")
    errors.append("Dependencies")

# Test 3: Check app.py exists
print("[3/7] Checking files...")
required_files = ['app.py', 'runner.py', 'requirements.txt']
for f in required_files:
    if os.path.exists(f):
        print(f"   ✓ {f}")
    else:
        print(f"   ✗ {f} missing")
        errors.append(f"Missing {f}")

# Test 4: Check database
print("[4/7] Checking database...")
if os.path.exists('alpha_junior.db'):
    try:
        conn = sqlite3.connect('alpha_junior.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in c.fetchall()]
        print(f"   ✓ Database exists with tables: {', '.join(tables)}")
        conn.close()
    except Exception as e:
        print(f"   ✗ Database error: {e}")
        errors.append("Database")
else:
    print("   ℹ Database will be created on first run")

# Test 5: Try importing app
print("[5/7] Checking application...")
try:
    from app import app
    print("   ✓ App imports successfully")
except Exception as e:
    print(f"   ✗ App import failed: {e}")
    errors.append("App import")

# Test 6: Check if server is running
print("[6/7] Checking server status...")
try:
    response = requests.get('http://localhost:5000/api/health', timeout=2)
    if response.status_code == 200:
        print(f"   ✓ Server running! {response.json()}")
    else:
        print(f"   ⚠ Server returned: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ℹ Server not running (start with: python runner.py)")
except Exception as e:
    print(f"   ⚠ Check error: {e}")

# Test 7: Check logs directory
print("[7/7] Checking logs...")
if not os.path.exists('logs'):
    os.makedirs('logs')
    print("   ✓ Created logs directory")
else:
    print("   ✓ Logs directory exists")

print()
print("="*60)
if errors:
    print(f"  ✗ ERRORS FOUND: {len(errors)}")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  ✅ ALL CHECKS PASSED!")
    print()
    print("  To start 24/7 server:")
    print("    python runner.py")
    print("  Or double-click: start_24_7_simple.bat")
    print()
    print("  Access: http://localhost:5000")
print("="*60)
