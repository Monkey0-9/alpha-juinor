import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


class BacktestRegistry:
    """
    Institutional-grade backtest registry.

    Principles:
    - Append-only
    - Immutable run artifacts
    - One directory per run
    - Central manifest for discovery
    """

    def __init__(self, base_dir: str = "output/backtests"):
        self.base_dir = Path(base_dir)
        self.manifest_path = self.base_dir / "manifest.json"

        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self.manifest_path.exists():
            self._init_manifest()

    # -------------------------
    # Internal helpers
    # -------------------------

    def _init_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump([], f, indent=2)

    def _load_manifest(self) -> List[Dict[str, Any]]:
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _write_manifest(self, manifest: List[Dict[str, Any]]):
        # Atomic-ish write
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        tmp_path.replace(self.manifest_path)

    # -------------------------
    # Public API
    # -------------------------

    def register_run(
        self,
        config: Dict[str, Any],
        results_df: pd.DataFrame,
        metrics: Dict[str, Any],
        extra_artifacts: Dict[str, pd.DataFrame] | None = None,
    ) -> str:
        """
        Register a completed backtest run.

        Saves:
        - config.json
        - equity.csv
        - optional additional artifacts
        - updates manifest.json

        Returns run_id.
        """
        run_id = uuid.uuid4().hex[:10]
        timestamp = datetime.utcnow().isoformat() + "Z"

        run_dir = self.base_dir / run_id
        run_dir.mkdir(exist_ok=False)

        # -------------------------
        # Save config
        # -------------------------
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        # -------------------------
        # Save primary result
        # -------------------------
        results_df.to_csv(run_dir / "equity.csv")

        # -------------------------
        # Save extra artifacts (optional)
        # -------------------------
        if extra_artifacts:
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir()
            for name, df in extra_artifacts.items():
                df.to_csv(artifacts_dir / f"{name}.csv")

        # -------------------------
        # Update manifest
        # -------------------------
        record = {
            "run_id": run_id,
            "timestamp": timestamp,
            "strategy": config.get("strategy", config.get("strategy_name", "unknown")),
            "metrics": {
                "final_equity": metrics.get("final_equity"),
                "total_return": metrics.get("total_return"),
                "annualized_vol": metrics.get("annualized_vol"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
            },
            "path": str(run_dir),
        }

        manifest = self._load_manifest()
        manifest.insert(0, record)  # newest first
        self._write_manifest(manifest)

        print(f"   [Registry] Run {run_id} registered.")
        return run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        """
        Return list of all registered runs (metadata only).
        """
        return self._load_manifest()

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """
        Load a full run (metadata + equity curve).
        """
        manifest = self._load_manifest()
        for r in manifest:
            if r["run_id"] == run_id:
                run_dir = Path(r["path"])
                equity = pd.read_csv(run_dir / "equity.csv", index_col=0, parse_dates=True)

                return {
                    "meta": r,
                    "equity": equity,
                }

        raise KeyError(f"Run {run_id} not found in registry.")
