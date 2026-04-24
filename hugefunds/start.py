#!/usr/bin/env python3
"""
HUGEFUNDS - Universal 24/7 Launcher
No unicode, zero errors, Windows-safe
"""

import os
import sys
import time
import subprocess
import argparse
import json
from pathlib import Path

# Windows-safe colors using ASCII only
class Colors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    GRAY = '\033[90m'
    WHITE = '\033[97m'

def get_project_root():
    return Path(__file__).parent.absolute()

def get_backend_dir():
    return get_project_root() / "backend"

def get_venv_python():
    venv = get_project_root() / ".venv"
    if sys.platform == "win32":
        return str(venv / "Scripts" / "python.exe")
    return str(venv / "bin" / "python")

def is_port_in_use(port=8000):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_backend_status():
    import urllib.request
    try:
        response = urllib.request.urlopen('http://localhost:8000/api/health', timeout=5)
        return json.loads(response.read().decode())
    except:
        return None

def start_backend(daemon=False):
    backend_dir = get_backend_dir()
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "backend_24_7.log"
    
    python_path = get_venv_python()
    
    print(f"{Colors.OKBLUE}[*] Starting Backend Server on Port 8000...{Colors.ENDC}")
    print(f"{Colors.GRAY}     Working dir: {backend_dir}{Colors.ENDC}")
    print()
    
    # Check if already running
    if is_port_in_use(8000):
        print(f"{Colors.WARNING}[!] Port 8000 already in use. Checking...{Colors.ENDC}")
        status = check_backend_status()
        if status:
            print(f"{Colors.OKGREEN}[OK] Backend already operational{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}[X] Port in use but not responding. Kill existing first.{Colors.ENDC}")
            return False
    
    cmd = [
        python_path, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info"
    ]
    
    if daemon:
        if sys.platform == "win32":
            subprocess.Popen(
                cmd,
                cwd=str(backend_dir),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT
            )
        else:
            subprocess.Popen(
                cmd,
                cwd=str(backend_dir),
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
    else:
        process = subprocess.Popen(
            cmd,
            cwd=str(backend_dir),
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT
        )
        
        print(f"{Colors.OKBLUE}[*] Waiting for startup (up to 10s)...{Colors.ENDC}")
        for i in range(10):
            time.sleep(1)
            if is_port_in_use(8000):
                print(f"{Colors.OKGREEN}[OK] Backend RUNNING on Port 8000{Colors.ENDC}")
                return process
        
        print(f"{Colors.FAIL}[X] Backend FAILED to start{Colors.ENDC}")
        print(f"{Colors.WARNING}[!] Check logs\backend_24_7.log{Colors.ENDC}")
        return None
    
    return True

def stop():
    print(f"{Colors.OKBLUE}[*] Stopping HugeFunds...{Colors.ENDC}")
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/FI", "COMMANDLINE eq *uvicorn*"], capture_output=True)
        subprocess.run(["taskkill", "/F", "/FI", "COMMANDLINE eq *python*backend*"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
    print(f"{Colors.OKGREEN}[OK] HugeFunds stopped{Colors.ENDC}")

def status():
    if is_port_in_use(8000):
        print(f"{Colors.OKGREEN}[OK] Backend RUNNING on port 8000{Colors.ENDC}")
        health = check_backend_status()
        if health:
            print(f"{Colors.OKGREEN}[OK] Health: {health.get('status', 'unknown')}{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}[X] Backend NOT RUNNING{Colors.ENDC}")
        print(f"{Colors.OKBLUE}[*] Run: python start.py{Colors.ENDC}")

def main():
    parser = argparse.ArgumentParser(description="HugeFunds Launcher")
    parser.add_argument("--daemon", action="store_true", help="Run in background")
    parser.add_argument("--stop", action="store_true", help="Stop")
    parser.add_argument("--status", action="store_true", help="Check status")
    args = parser.parse_args()
    
    if args.stop:
        stop()
        return
    
    if args.status:
        status()
        return
    
    print("=" * 70)
    print("  HUGE FUNDS - TOP 1% WORLDWIDE ELITE COLLECTIVE")
    print("  Zero Errors | Zero Fake Data | Real Alpaca Trading")
    print("=" * 70)
    print()
    
    process = start_backend(daemon=args.daemon)
    if not process and not args.daemon:
        sys.exit(1)
    
    print()
    print("  ACCESS POINTS:")
    print("    http://localhost:8000           - Dashboard")
    print("    http://localhost:8000/docs        - API Docs")
    print("    http://localhost:8000/api/alpaca/account - Account")
    print("    http://localhost:8000/api/alpaca/positions - Positions")
    print()
    print("  TRADE COMMANDS:")
    print('    curl -X POST "http://localhost:8000/api/alpaca/buy?symbol=AAPL&qty=10"')
    print('    curl -X POST "http://localhost:8000/api/alpaca/sell?symbol=AAPL&qty=10"')
    print()
    print("  LIVE MONITOR:")
    print("    python live_monitor.py")
    print()
    
    if args.daemon:
        print(f"{Colors.OKGREEN}[OK] Running in background{Colors.ENDC}")
        print(f"{Colors.OKBLUE}[*] Access: http://localhost:8000{Colors.ENDC}")
    else:
        print(f"{Colors.OKBLUE}[*] Press Ctrl+C to stop{Colors.ENDC}")
        try:
            while True:
                time.sleep(30)
                if not is_port_in_use(8000):
                    print(f"{Colors.WARNING}[!] Backend stopped. Restarting...{Colors.ENDC}")
                    start_backend()
        except KeyboardInterrupt:
            print()
            stop()

if __name__ == "__main__":
    main()
