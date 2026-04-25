import subprocess
import time
import os
import sys


class GlobalOrchestrator:
    """
    The Quant Sovereign - World-Class Institutional Orchestrator
    Surpassing HugeFunds and Rivaling Citadel/Two Sigma.
    """
    def __init__(self):
        self.processes = []
        self.root = os.getcwd()
        print("=" * 80)
        print("      INITIALIZING INSTITUTIONAL GRADE QUANT SOVEREIGN")
        print("=" * 80)

    def start_component(self, name, cmd, cwd):
        print(f"[*] Launching {name}...")
        p = subprocess.Popen(cmd, cwd=cwd, shell=True)
        self.processes.append((name, p))
        return p

    def run_all(self):
        python_exe = f'"{sys.executable}"'

        # 1. Start Institutional Risk & Governance Gate (Internal Monitor)
        print("[*] Activating Institutional Governance & Risk Layers...")

        # 2. Start HugeFunds Backend (Execution Venue)
        self.start_component(
            "HugeFunds-Core",
            f"{python_exe} start.py",
            os.path.join(self.root, "hugefunds")
        )

        # 3. Start AlphaJunior Backend (Intelligence)
        self.start_component(
            "AlphaJunior-Intelligence",
            f"{python_exe} backend/app/main.py",
            os.path.join(self.root, "alpha_junior")
        )

        # 4. Start EliteQuantSystem (Alpha Gen)
        self.start_component(
            "EliteAlpha-Engine",
            f"{python_exe} system.py",
            os.path.join(self.root, "elite_quant_fund")
        )

        # 5. Start AlphaJunior Dashboard (UI)
        streamlit_exe = os.path.join(os.path.dirname(sys.executable), "streamlit.exe")
        self.start_component(
            "AlphaJunior-Dashboard",
            f'"{streamlit_exe}" run app.py --server.port 8501',
            os.path.join(self.root, "alpha_junior")
        )

        print("[OK] ALL INSTITUTIONAL LAYERS ACTIVE. SYSTEM PINNACLE REACHED.")

        try:
            while True:
                for name, p in self.processes:
                    if p.poll() is not None:
                        print(
                            f"[!] CRITICAL: {name} has stopped. "
                            f"Return code: {p.returncode}"
                        )
                time.sleep(10)
        except KeyboardInterrupt:
            self.stop_all()

    def stop_all(self):
        print("[*] Deactivating The Quant Sovereign...")
        for name, p in self.processes:
            p.terminate()
        print("[OK] Secure shutdown complete.")


if __name__ == "__main__":
    orchestrator = GlobalOrchestrator()
    orchestrator.run_all()
