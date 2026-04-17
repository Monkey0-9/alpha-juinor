"""
Project Setup Script - MiniQuantFund v4.0.0
Automated setup for paper trading system
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(command, description):
    """Run command and handle errors"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"SUCCESS: {description}")
            if result.stdout:
                print(f"OUTPUT: {result.stdout}")
        else:
            print(f"ERROR: {description}")
            print(f"ERROR OUTPUT: {result.stderr}")
            return False
    except Exception as e:
        print(f"EXCEPTION: {description} - {e}")
        return False
    
    return True

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"Python version OK: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"Python version too old: {version.major}.{version.minor}.{version.micro}")
        print("Please install Python 3.8 or higher")
        return False

def install_dependencies():
    """Install required packages"""
    packages = [
        "alpaca-trade-api",
        "pandas", 
        "numpy",
        "asyncio",
        "aiohttp",
        "prometheus-client",
        "grafana-api",
        "psycopg2-binary",
        "redis",
        "python-dotenv",
        "websockets",
        "requests",
        "scipy",
        "matplotlib",
        "seaborn"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("ERROR: .env file not found!")
        return False
    
    # Load and check environment variables
    with open(env_file, 'r') as f:
        env_content = f.read()
    
    required_vars = [
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY"
    ]
    
    for var in required_vars:
        if var not in env_content:
            print(f"ERROR: {var} not found in .env file")
            return False
    
    print("Environment variables OK")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data",
        "backups",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")
    
    return True

def setup_database():
    """Setup database (basic check)"""
    try:
        import psycopg2
        print("PostgreSQL driver available")
        return True
    except ImportError:
        print("PostgreSQL driver not installed")
        return False

def main():
    """Main setup function"""
    print("MiniQuantFund v4.0.0 - Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        return False
    
    # Setup environment
    if not setup_environment():
        print("Environment setup failed")
        return False
    
    # Create directories
    if not create_directories():
        print("Failed to create directories")
        return False
    
    # Setup database
    if not setup_database():
        print("Database setup failed")
        return False
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE! Ready to run paper trading system")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run: python run_paper_trading_system.py")
    print("2. Monitor: tail -f logs/paper_trading.log")
    print("3. Check: curl http://localhost:8080/health")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
