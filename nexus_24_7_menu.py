#!/usr/bin/env python3
"""
Quick 24/7 Startup Script
Provides simple commands to start/stop 24/7 trading operation
"""

import subprocess
import sys
import os
from pathlib import Path

def print_menu():
    print("\n" + "=" * 70)
    print("NEXUS 24/7 TRADING PLATFORM - QUICK STARTUP")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Start (Command Line)     - python nexus_24_7.py")
    print("  2. Install Task Scheduler   - Setup for permanent 24/7 operation")
    print("  3. Check Task Status        - View scheduled task status")
    print("  4. Stop Task Scheduler      - Pause 24/7 operation")
    print("  5. Remove Task Scheduler    - Uninstall scheduled task")
    print("  6. View Logs                - Show recent log entries")
    print("  7. Run Verification         - Check system status")
    print("  8. View Configuration       - Show current settings")
    print("  9. Exit")
    print("\n" + "=" * 70 + "\n")

def start_command_line():
    """Start nexus_24_7.py directly."""
    print("Starting Nexus 24/7 in command line mode...")
    print("(Press Ctrl+C to stop)\n")
    try:
        subprocess.run([sys.executable, "nexus_24_7.py"])
    except KeyboardInterrupt:
        print("\nStopped by user")

def install_task_scheduler():
    """Install Windows Task Scheduler task."""
    print("Installing Windows Task Scheduler task...")
    print("(Requires Administrator privileges)\n")
    try:
        subprocess.run(
            ["PowerShell", "-ExecutionPolicy", "Bypass", "-File", 
             "setup_scheduled_task.ps1", "-Action", "install"],
            check=True
        )
        print("\n[OK] Task installed successfully!")
        print("It will start automatically on system reboot.")
        print("To start immediately: Start-ScheduledTask -TaskName 'Nexus24x7TradingPlatform'")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Failed to install task: {e}")
    except FileNotFoundError:
        print("\n[FAIL] PowerShell not found or not in PATH")

def check_task_status():
    """Check scheduled task status."""
    print("Checking task status...\n")
    try:
        subprocess.run(
            ["PowerShell", "-ExecutionPolicy", "Bypass", "-File", 
             "setup_scheduled_task.ps1", "-Action", "status"],
            check=False
        )
    except FileNotFoundError:
        print("PowerShell not found or not in PATH")

def stop_task_scheduler():
    """Stop the scheduled task."""
    print("Stopping Windows Task Scheduler task...")
    try:
        subprocess.run(
            ["PowerShell", "-Command", 
             "Stop-ScheduledTask -TaskName 'Nexus24x7TradingPlatform' -ErrorAction SilentlyContinue"],
            check=False
        )
        print("[OK] Task stopped successfully!")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

def remove_task_scheduler():
    """Remove the scheduled task."""
    response = input("Are you sure you want to remove the scheduled task? (y/n): ")
    if response.lower() != 'y':
        return
    
    print("Removing Windows Task Scheduler task...")
    print("(Requires Administrator privileges)\n")
    try:
        subprocess.run(
            ["PowerShell", "-ExecutionPolicy", "Bypass", "-File", 
             "setup_scheduled_task.ps1", "-Action", "remove"],
            check=False
        )
        print("\n[OK] Task removed successfully!")
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")

def view_logs():
    """View recent log entries."""
    log_dir = Path("logs")
    if not log_dir.exists():
        print("No logs directory found yet.")
        return
    
    log_files = sorted(log_dir.glob("nexus_24_7_*.log"), reverse=True)
    if not log_files:
        print("No log files found yet.")
        return
    
    latest_log = log_files[0]
    print(f"Viewing latest log file: {latest_log.name}\n")
    print("=" * 70)
    
    try:
        # Read last 50 lines
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line.rstrip())
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    print("=" * 70)

def run_verification():
    """Run production readiness verification."""
    print("Running production readiness verification...\n")
    try:
        subprocess.run([sys.executable, "verify_production_ready.py"])
    except FileNotFoundError:
        print("Verification script not found")

def view_configuration():
    """Display current configuration."""
    print("Current Configuration:\n")
    print("=" * 70)
    
    try:
        from nexus.utils.config import Config
        
        config_items = {
            "API Host": Config.API_HOST,
            "API Port": Config.API_PORT,
            "Streamlit Port": Config.STREAMLIT_PORT,
            "Backend URL": Config.BACKEND_URL,
            "Max Position Size": f"{Config.MAX_POSITION_SIZE * 100:.0f}%",
            "Max Drawdown": f"{Config.MAX_DRAWDOWN * 100:.0f}%",
            "Max Open Positions": Config.MAX_OPEN_POSITIONS,
            "Max Daily Trades": Config.MAX_DAILY_TRADES,
            "Min Order USD": f"${Config.MIN_ORDER_USD:,.0f}",
            "Candidate Pool Size": Config.CANDIDATE_POOL_SIZE,
            "Paper Trading": Config.ALPACA_PAPER,
        }
        
        for key, value in config_items.items():
            print(f"  {key:.<30} {value}")
        
        print("=" * 70)
        
        # Check credentials
        valid, missing = Config.validate()
        if valid:
            print("\n[OK] Credentials validated successfully")
        else:
            print(f"\n[FAIL] Missing credentials: {missing}")
            
    except Exception as e:
        print(f"Error loading configuration: {e}")

def main():
    """Main menu loop."""
    while True:
        print_menu()
        choice = input("Select option (1-9): ").strip()
        
        if choice == '1':
            start_command_line()
        elif choice == '2':
            install_task_scheduler()
        elif choice == '3':
            check_task_status()
        elif choice == '4':
            stop_task_scheduler()
        elif choice == '5':
            remove_task_scheduler()
        elif choice == '6':
            view_logs()
        elif choice == '7':
            run_verification()
        elif choice == '8':
            view_configuration()
        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid option, please try again")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
