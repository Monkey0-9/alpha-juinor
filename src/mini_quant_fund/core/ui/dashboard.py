
import os
from datetime import datetime
from abc import ABC, abstractmethod

class BaseUI(ABC):
    @abstractmethod
    def update(self, state: dict):
        pass

class NullUI(BaseUI):
    """Silent UI for headless/production runs."""
    def update(self, state: dict):
        pass

class TerminalDashboard(BaseUI):
    """Institutional terminal dashboard with safe rendering."""
    
    def __init__(self):
        self.last_update = None
        
    def _clear_screen(self):
        if os.name == "nt":
            os.system("cls")
        else:
            print("\033[2J\033[H", end="")

    def update(self, state: dict):
        now = datetime.utcnow()
        if self.last_update and (now - self.last_update).total_seconds() < 1.0:
            return
            
        self._clear_screen()
        print("=" * 80)
        print(f" QUANT FUND OS | {now.strftime('%Y-%m-%d %H:%M:%S')} UTC ")
        print("=" * 80)
        print(f" Status: {state.get('status', 'N/A')}")
        print(f" Cycle:  {state.get('cycle_count', 0)}")
        print(f" P&L:    {state.get('total_pnl', 0.0):.2f}")
        print("-" * 80)
        
        positions = state.get("positions", {})
        if positions:
            print(f" {'SYMBOL':<8} | {'QTY':>10} | {'PRICE':>10} | {'PNL':>12}")
            for s, p in positions.items():
                # Handling both dict and object style for compatibility during migration
                qty = getattr(p, 'quantity', p.get('quantity', 0))
                price = getattr(p, 'current_price', p.get('current_price', 0))
                pnl = getattr(p, 'unrealized_pnl', p.get('unrealized_pnl', 0))
                print(f" {s:<8} | {qty:>10.2f} | {price:>10.2f} | {pnl:>12.2f}")
        else:
            print(" No active positions.")
        
        print("=" * 80)
        self.last_update = now
