"""
monitor/cli_renderer.py
P2-1: Operator CLI Dashboard - Human-readable status display
"""
import time
from datetime import datetime
from typing import Dict, Any
import logging

# Try to use colorama for cross-platform colors
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback - no colors
    class Fore:
        GREEN = RED = YELLOW = CYAN = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = ""

logger = logging.getLogger(__name__)


class CLIRenderer:
    """
    Human-readable CLI dashboard for operators.

    Updates every 5-10 seconds with aggregated system status.
    """

    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.last_update = 0

    def render_dashboard(self, metrics: Dict[str, Any]):
        """
        Render dashboard with color-coded status.

        Args:
            metrics: Dict containing:
                - uptime_sec
                - symbols_count
                - cycles
                - model_errors
                - arima_fallbacks
                - active_positions
                - last_trade
                - ml_state
                - system_state
                - latency_p50
                - latency_p95
        """
        now = time.time()

        # Throttle updates
        if now - self.last_update < self.update_interval:
            return

        self.last_update = now

        # Clear screen (platform-independent way)
        print("\033[2J\033[H", end="")

        # Header
        print("=" * 80)
        print(f"{Style.BRIGHT}MINI QUANT FUND - OPERATOR DASHBOARD{Style.RESET_ALL}")
        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()

        # System Status Section
        state = metrics.get("system_state", "UNKNOWN")
        state_color = self._get_state_color(state)

        print(f"{Style.BRIGHT}SYSTEM STATUS:{Style.RESET_ALL}")
        print(f"  State:        {state_color}{state}{Style.RESET_ALL}")
        print(f"  Uptime:       {self._format_uptime(metrics.get('uptime_sec', 0))}")
        print(f"  Cycles:       {metrics.get('cycles', 0)}")
        print(f"  Symbols:      {metrics.get('symbols_count', 0)}")
        print()

        # ML Status Section
        ml_state = metrics.get("ml_state", "UNKNOWN")
        ml_color = self._get_ml_state_color(ml_state)

        print(f"{Style.BRIGHT}ML ALPHA STATUS:{Style.RESET_ALL}")
        print(f"  Mode:         {ml_color}{ml_state}{Style.RESET_ALL}")
        print(f"  Errors:       {self._color_metric(metrics.get('model_errors', 0), [0, 10, 50], 'errors')}")
        print(f"  ARIMA FB:     {self._color_metric(metrics.get('arima_fallbacks', 0), [0, 5, 20], 'fallbacks')}")
        print()

        # Trading Status Section
        print(f"{Style.BRIGHT}TRADING STATUS:{Style.RESET_ALL}")
        print(f"  Positions:    {metrics.get('active_positions', 0)}")
        print(f"  Last Trade:   {metrics.get('last_trade', 'N/A')}")
        print()

        # Performance Section
        latency_p50 = metrics.get('latency_p50', 0)
        latency_p95 = metrics.get('latency_p95', 0)

        print(f"{Style.BRIGHT}PERFORMANCE:{Style.RESET_ALL}")
        print(f"  Latency P50:  {self._color_latency(latency_p50)} ms")
        print(f"  Latency P95:  {self._color_latency(latency_p95)} ms")
        print()

        # Footer
        print("=" * 80)
        print(f"{Fore.CYAN}Press Ctrl+C to stop{Style.RESET_ALL}")
        print("=" * 80)

    def _get_state_color(self, state: str) -> str:
        """Get color for system state."""
        if not COLORS_AVAILABLE:
            return ""

        if state == "OK":
            return Fore.GREEN + Style.BRIGHT
        elif state == "DEGRADED":
            return Fore.YELLOW + Style.BRIGHT
        elif state == "HALTED":
            return Fore.RED + Style.BRIGHT
        return Fore.WHITE

    def _get_ml_state_color(self, ml_state: str) -> str:
        """Get color for ML state."""
        if not COLORS_AVAILABLE:
            return ""

        if "ENABLED|OK" in ml_state:
            return Fore.GREEN + Style.BRIGHT
        elif "ENABLED|DEGRADED" in ml_state:
            return Fore.YELLOW + Style.BRIGHT
        elif "DISABLED" in ml_state:
            return Fore.RED + Style.BRIGHT
        return Fore.WHITE

    def _color_metric(self, value: int, thresholds: list, label: str) -> str:
        """Color-code metric based on thresholds [low, medium, high]."""
        if not COLORS_AVAILABLE:
            return str(value)

        low, medium, high = thresholds

        if value <= low:
            color = Fore.GREEN
        elif value <= medium:
            color = Fore.YELLOW
        else:
            color = Fore.RED

        return f"{color}{value}{Style.RESET_ALL}"

    def _color_latency(self, latency_ms: float) -> str:
        """Color-code latency."""
        if not COLORS_AVAILABLE:
            return f"{latency_ms:.1f}"

        if latency_ms < 100:
            color = Fore.GREEN
        elif latency_ms < 500:
            color = Fore.YELLOW
        else:
            color = Fore.RED

        return f"{color}{latency_ms:.1f}{Style.RESET_ALL}"

    def _format_uptime(self, seconds: int) -> str:
        """Format uptime in human-readable form."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"


# Global renderer instance
cli_renderer = CLIRenderer(update_interval=5)
