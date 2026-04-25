import subprocess
import sys
import os
import time

def check_dependencies():
    print("[*] Checking system dependencies...")
    # Add dependency checks here
    pass

def run():
    print("="*80)
    print("      MINI QUANT FUND - COMPLETE INSTITUTIONAL SYSTEM")
    print("="*80)
    
    # Launch the orchestrator
    orchestrator_path = os.path.join("src", "mini_quant_fund", "orchestration", "orchestrator.py")
    
    try:
        subprocess.run([sys.executable, orchestrator_path])
    except KeyboardInterrupt:
        print("\n[!] System interrupted by user.")

if __name__ == "__main__":
    run()
