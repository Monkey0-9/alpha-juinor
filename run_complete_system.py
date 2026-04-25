import subprocess
import sys
import os
import time
import signal

def kill_port(port):
    """Forcefully kills any process sitting on the specified port (Windows)."""
    try:
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        for line in output.strip().split('\n'):
            parts = line.split()
            if len(parts) > 4:
                pid = parts[-1]
                print(f"[*] Killing process {pid} on port {port}...")
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
    except:
        pass

def run():
    print("="*80)
    print("      CELESTIAL SOVEREIGN - APEX INSTITUTIONAL QUANT")
    print("      [SINGULARITY V4.0 - OMEGA STATE]")
    print("="*80)
    
    print("[*] Purging port collisions...")
    for port in [8000, 8001, 8501]:
        kill_port(port)
        
    print("[*] Activating Institutional Governance...")
    time.sleep(1)
    
    # Launch the orchestrator
    orchestrator_path = os.path.join("src", "mini_quant_fund", "orchestration", "orchestrator.py")
    
    print("\n" + "="*80)
    print("  SYSTEM PINNACLE REACHED. LAUNCHING OMNISCIENCE LOOP.")
    print("="*80 + "\n")
    
    try:
        subprocess.run([sys.executable, orchestrator_path])
    except KeyboardInterrupt:
        print("\n[!] Sovereign Shutdown Initiated.")
        # Cleanup
        for port in [8000, 8001, 8501]:
            kill_port(port)
        print("[OK] Secure exit complete.")

if __name__ == "__main__":
    run()
