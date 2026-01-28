"""
ui/terminal_dashboard.py

Section H: Observability (Institutional Upgrade).
Uses Rich for structured PM-centric dashboard.
"""
from datetime import datetime
from typing import Dict, Any, List

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TerminalDashboard:
    def __init__(self):
        self.console = Console()
        self.state = {
            "portfolio": {"nav": 0.0, "cvar_95": 0.0, "exposure": 0.0, "pnl_bps": 0.0},
            "data_health": {
                "ok_count": 0, "degraded": [], "quarantined": [], "avg_quality": 1.0
            },
            "regime": {"status": "NORMAL", "confidence": 1.0, "override": None},
            "competition": {"candidates": [], "allocated_count": 0, "rejected_count": 0},
            "execution": {"blocked_count": 0, "executed_count": 0, "active_orders": 0},
            "alerts": []
        }

    def update(self, key: str, data: Dict[str, Any]):
        """Update dashboard state section."""
        if key in self.state:
            if isinstance(self.state[key], dict):
                self.state[key].update(data)
            elif isinstance(self.state[key], list):
                self.state[key] = data

    def add_alert(self, code: str, msg: str):
        """Add structured alert."""
        self.state["alerts"].append({
            "ts": datetime.utcnow().strftime("%H:%M:%S"),
            "code": code,
            "msg": msg
        })
        self.state["alerts"] = self.state["alerts"][-5:] # Keep last 5

    def update_competition(self, results: List[Any]):
        """Update capital competition stats from CompetitionResult objects."""
        # Convert objects to simpler dicts for display
        candidates = []
        allocated = 0
        rejected = 0
        for r in results:
            candidates.append({
                "rank": getattr(r, 'rank', 0),
                "symbol": getattr(r, 'symbol', '?'),
                "score": getattr(r, 'score', 0.0),
                "decision": getattr(r, 'decision', 'UNKNOWN'),
                "weight": getattr(r, 'weight', 0.0),
                "reason": getattr(r, 'reason', '')
            })
            if getattr(r, 'decision', '') == 'ALLOCATE':
                allocated += 1
            elif getattr(r, 'decision', '') == 'REJECT':
                rejected += 1

        self.state["competition"] = {
            "candidates": candidates[:10], # Top 10 only for display
            "allocated_count": allocated,
            "rejected_count": rejected,
            "total_candidates": len(results)
        }

    def _make_data_panel(self) -> Panel:
        d = self.state["data_health"]
        ok = d.get("ok_count", 0)
        deg = len(d.get("degraded", []))
        quar = len(d.get("quarantined", []))
        qual = d.get("avg_quality", 1.0)

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(f"OK: [green]{ok}[/]")
        grid.add_row(f"Degraded: [yellow]{deg}[/]")
        grid.add_row(f"Invalid: [red]{quar}[/]")
        grid.add_row(f"quality_score: {qual:.2f}")

        return Panel(grid, title="[DATA HEALTH]", border_style="blue")

    def _make_regime_panel(self) -> Panel:
        r = self.state["regime"]
        st = r.get("status", "NORMAL")
        conf = r.get("confidence", 0.0)
        ovr = r.get("override")

        color = "green"
        if st in ["RISK_OFF", "STRESS"]: color = "yellow"
        if st == "CRISIS": color = "red bold"

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(f"State: [{color}]{st}[/]")
        grid.add_row(f"Conf: {conf:.0%}")
        if ovr:
            grid.add_row(f"Limit: [magenta]{ovr}[/]")
        else:
            grid.add_row("Limit: [dim]None[/]")

        return Panel(grid, title="[REGIME]", border_style="cyan")

    def _make_portfolio_panel(self) -> Panel:
        p = self.state["portfolio"]

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(f"NAV: [bold white]${p.get('nav', 0):,.0f}[/]")
        grid.add_row(f"Gross: {p.get('exposure', 0):.1%}")
        grid.add_row(f"CVaR95: [red]{p.get('cvar_95', 0):.1%}[/]")

        return Panel(grid, title="[PORTFOLIO]", border_style="white")

    def _make_competition_table(self) -> Table:
        c = self.state["competition"]
        t = Table(show_edge=False, show_header=True, expand=True, box=None)
        t.add_column("Rk", width=2, justify="right")
        t.add_column("Sym", width=5)
        t.add_column("Score", justify="right")
        t.add_column("Dec", justify="center")
        t.add_column("Reason", style="dim")

        for cand in c.get("candidates", []):
            dec_color = "green" if cand["decision"] == "ALLOCATE" else "red"
            t.add_row(
                str(cand["rank"]),
                cand["symbol"],
                f"{cand['score']:.4f}",
                f"[{dec_color}]{cand['decision'][0:4]}[/]",
                cand["reason"][:20]
            )

        return t

    def _make_ml_panel(self) -> Panel:
        ml = self.state.get("ml", {"health": 0.0, "active_models": 0, "arima_fb": 0})
        health = ml.get("health", 0.0)
        active = ml.get("active_models", 0)
        fb = ml.get("arima_fb", 0)

        color = "green"
        if health < 0.8: color = "yellow"
        if health < 0.5: color = "red bold"

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_row(f"ML Health: [{color}]{health:.1%}[/]")
        grid.add_row(f"Active Models: {active}")
        grid.add_row(f"ARIMA Fallbacks: {fb}")

        return Panel(grid, title="[ML INTELLIGENCE]", border_style="magenta")

    def render(self):

        """Render the full dashboard."""

        # Top Row: Health & Stats
        top_layout = Layout()
        top_layout.split_row(
            Layout(self._make_data_panel(), ratio=1),
            Layout(self._make_regime_panel(), ratio=1),
            Layout(self._make_ml_panel(), ratio=1),
            Layout(self._make_portfolio_panel(), ratio=1),
            Layout(self._make_exec_panel(), ratio=1)
        )


        # Middle: Logic/Competition
        mid_panel = Panel(
            self._make_competition_table(),
            title="[CAPITAL COMPETITION]",
            border_style="magenta"
        )

        # Bottom: Alerts
        alerts_txt = Text()
        for a in self.state["alerts"]:
            alerts_txt.append(f"{a['ts']} [{a['code']}] {a['msg']}\n", style="white")

        bottom_panel = Panel(alerts_txt, title="[ALERTS]", border_style="red", height=8)

        # Combine
        full_layout = Layout()
        full_layout.split_column(
            Layout(top_layout, size=6),
            Layout(mid_panel, ratio=2),
            Layout(bottom_panel, size=8)
        )

        print("\n")
        self.console.print(full_layout)
        print("\n")

dashboard = TerminalDashboard()

