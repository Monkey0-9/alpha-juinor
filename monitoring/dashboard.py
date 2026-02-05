import time
import random
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def render_dashboard():
    """
    Simple console dashboard reading from logs (simulated reading here).
    """
    print("="*60)
    print("      GLOBAL FINANCIAL AI - LIVE MONITORING DASHBOARD      ")
    print("="*60)

    # 1. PnL
    live_pnl = random.uniform(-0.5, 1.2)
    backtest_corr = random.uniform(0.88, 0.94)
    print(f"[PERFORMANCE]")
    print(f"  Daily PnL:       {live_pnl:+.2f}%")
    print(f"  Vs Backtest:     {backtest_corr:.2f} (Target > 0.85) [PASS]")

    # 2. RL Allocation
    print("-" * 60)
    print(f"[RL META-CONTROLLER]")
    regime = "HIGH_VOL" if random.random() < 0.2 else "NORMAL"
    print(f"  Market Regime:   {regime}")
    print(f"  Allocation:      [Defensive: 10% | Balanced: 60% | Aggressive: 30%]")

    # 3. Network Alpha
    print("-" * 60)
    print(f"[NETWORK ALPHA]")
    print(f"  Active Signals:  {random.randint(0, 3)}")
    print(f"  Top Signal:      AAPL (Cluster Buy, Score: 0.82)")
    print(f"  Decay Status:    Fresh (Day 1/20)")

    print("="*60)
    print("Press Ctrl+C to exit.")

if __name__ == "__main__":
    try:
        while True:
            clear_screen()
            render_dashboard()
            time.sleep(2)
    except KeyboardInterrupt:
        print("Dashboard closed.")
