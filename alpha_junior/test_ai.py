#!/usr/bin/env python3
"""
Alpha Junior - AI System Test
Verifies AI Brain and Autonomous Trader are working
"""

import os
import sys

print("=" * 70)
print("  🤖 ALPHA JUNIOR - AI SYSTEM VERIFICATION")
print("=" * 70)
print()

errors = []
warnings = []

# Test 1: Check Python
print("[1/8] Checking Python version...")
version = sys.version_info
if version.major >= 3 and version.minor >= 11:
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"   ⚠ Python {version.major}.{version.minor} (3.11+ recommended)")
    warnings.append("Python version")

# Test 2: Check dependencies
print("[2/8] Checking dependencies...")
try:
    import flask
    print("   ✓ Flask installed")
except:
    print("   ✗ Flask missing - run: pip install flask")
    errors.append("Flask")

try:
    import flask_cors
    print("   ✓ Flask-CORS installed")
except:
    print("   ✗ Flask-CORS missing - run: pip install flask-cors")
    errors.append("Flask-CORS")

try:
    import requests
    print("   ✓ Requests installed")
except:
    print("   ✗ Requests missing - run: pip install requests")
    errors.append("Requests")

try:
    import numpy as np
    print("   ✓ NumPy installed")
except:
    print("   ✗ NumPy missing - run: pip install numpy")
    errors.append("NumPy")

# Test 3: Check AI files
print("[3/8] Checking AI system files...")
ai_files = ['brain.py', 'autonomous_trader.py', 'app.py', 'runner.py']
for f in ai_files:
    if os.path.exists(f):
        print(f"   ✓ {f}")
    else:
        print(f"   ✗ {f} missing")
        errors.append(f"Missing {f}")

# Test 4: Check .env
print("[4/8] Checking environment configuration...")
if os.path.exists('.env'):
    print("   ✓ .env file exists")
    
    # Check for Alpaca keys
    with open('.env', 'r') as f:
        env_content = f.read()
    
    if 'ALPACA_API_KEY=' in env_content:
        if 'YOUR_SECRET_KEY_HERE' in env_content:
            print("   ⚠ Alpaca API keys not configured (using placeholder)")
            warnings.append("Alpaca API keys not set")
        else:
            print("   ✓ Alpaca API keys configured")
    else:
        print("   ✗ ALPACA_API_KEY not found in .env")
        errors.append("ALPACA_API_KEY missing")
else:
    print("   ✗ .env file not found")
    errors.append(".env file missing")

# Test 5: Try importing AI Brain
print("[5/8] Testing AI Brain module...")
try:
    from brain import AlphaBrain, get_brain
    print("   ✓ AI Brain module imports successfully")
    
    # Check if we can create a brain instance (without API keys for now)
    brain = get_brain('test_key', 'test_secret')
    print(f"   ✓ Brain initialized with {len(brain.stock_universe)} stocks in universe")
    
except Exception as e:
    print(f"   ✗ Brain import failed: {e}")
    errors.append(f"Brain import: {e}")

# Test 6: Try importing Autonomous Trader
print("[6/8] Testing Autonomous Trader module...")
try:
    from autonomous_trader import AutonomousTrader, get_trader
    print("   ✓ Autonomous Trader module imports successfully")
except Exception as e:
    print(f"   ✗ Trader import failed: {e}")
    errors.append(f"Trader import: {e}")

# Test 7: Check database
print("[7/8] Checking database...")
try:
    import sqlite3
    if os.path.exists('alpha_junior.db'):
        conn = sqlite3.connect('alpha_junior.db')
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in c.fetchall()]
        print(f"   ✓ Database exists with tables: {', '.join(tables)}")
        conn.close()
    else:
        print("   ℹ Database will be created on first run")
except Exception as e:
    print(f"   ✗ Database error: {e}")
    errors.append(f"Database: {e}")

# Test 8: Check if server is running
print("[8/8] Checking server status...")
try:
    import requests
    response = requests.get('http://localhost:5000/api/health', timeout=2)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Server running! {data}")
    else:
        print(f"   ⚠ Server returned status {response.status_code}")
        warnings.append("Server may need restart")
except requests.exceptions.ConnectionError:
    print("   ℹ Server not running (start with: python runner.py)")
except Exception as e:
    print(f"   ⚠ Could not check server: {e}")

print()
print("=" * 70)

# Summary
if errors:
    print(f"  ❌ ERRORS FOUND: {len(errors)}")
    for e in errors:
        print(f"     - {e}")
    print()
    print("  Fix errors before running AI system")
    sys.exit(1)
elif warnings:
    print(f"  ⚠ WARNINGS: {len(warnings)}")
    for w in warnings:
        print(f"     - {w}")
    print()
    print("  ✅ AI System ready (with warnings)")
    print()
    print("  To start:")
    print("    1. Add Alpaca API keys to .env")
    print("    2. Run: .\\run_ai_autonomous.bat")
    print("    3. Start AI: curl -X POST http://localhost:5000/api/autonomous/start")
    sys.exit(0)
else:
    print("  ✅ ALL CHECKS PASSED!")
    print()
    print("  AI System is ready to run!")
    print()
    print("  Next steps:")
    print("    1. Run: .\\run_ai_autonomous.bat")
    print("    2. Start AI: curl -X POST http://localhost:5000/api/autonomous/start")
    print("    3. Watch: Dashboard and http://localhost:5000")
    print()
    print("  The AI Brain will scan 100+ stocks and trade automatically!")

print("=" * 70)
