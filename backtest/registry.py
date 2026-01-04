import json
import uuid
import sys
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import os

import pandas as pd


class BacktestRegistry:
    """
    Institutional-grade backtest registry with immutable artifacts.

    Principles:
    - Append-only (never overwrite)
    - Immutable run artifacts with checksums
    - Git hash tracking for reproducibility
    - One directory per run with unique ID
    - Central manifest for discovery
    
    Each run saves:
    - config.json (strategy configuration)
    - equity.csv (equity curve)
    - trades.csv (all trades)
    - data_manifest.json (data sources)
    - meta.json (git hash, checksums, Python version)
    - requirements.txt (frozen dependencies)
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
        # Atomic write
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        tmp_path.replace(self.manifest_path)

    def _sha256_of_file(self, path: Path) -> str:
        """Compute SHA256 checksum of file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _atomic_write_text(self, path: Path, text: str, encoding="utf-8"):
        """Atomic file write for text."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".txt")
        os.close(fd)
        try:
            with open(tmp, "w", encoding=encoding) as f:
                f.write(text)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def _atomic_write_df(self, path: Path, df: pd.DataFrame):
        """Atomic file write for DataFrame."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".csv")
        os.close(fd)
        try:
            df.to_csv(tmp, index=False)
            with open(tmp, "rb") as f:
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def _get_git_hash(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None

    def _freeze_requirements(self) -> str:
        """Get frozen pip requirements."""
        try:
            return subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"],
                stderr=subprocess.DEVNULL
            ).decode()
        except Exception:
            return ""

    # -------------------------
    # Public API
    # -------------------------

    def create_run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new run directory with unique ID.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        unique_suffix = uuid.uuid4().hex[:8]
        run_id = f"run_{timestamp}_{unique_suffix}"
        
        run_dir = self.base_dir / run_id
        
        if run_dir.exists():
            raise RuntimeError(f"Run directory {run_id} already exists.")
        
        run_dir.mkdir(parents=True, exist_ok=False)
        
        return {
            "run_id": run_id,
            "path": str(run_dir),
        }

    def save_artifacts(
        self,
        run_id: str,
        config: Dict[str, Any],
        equity_df: pd.DataFrame,
        trades_df: Optional[pd.DataFrame] = None,
        data_manifest: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save all required artifacts for a run with checksums and metadata.
        """
        run_dir = self.base_dir / run_id
        
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_id} does not exist.")
        
        config_path = run_dir / "config.json"
        self._atomic_write_text(config_path, json.dumps(config, indent=2, default=str))
        
        equity_path = run_dir / "equity.csv"
        self._atomic_write_df(equity_path, equity_df)
        
        trades_path = run_dir / "trades.csv"
        if trades_df is not None and not trades_df.empty:
            self._atomic_write_df(trades_path, trades_df)
        else:
            pd.DataFrame().to_csv(trades_path, index=False)
        
        data_manifest_path = run_dir / "data_manifest.json"
        manifest_data = data_manifest or {}
        self._atomic_write_text(
            data_manifest_path,
            json.dumps(manifest_data, indent=2, default=str)
        )
        
        req_path = run_dir / "requirements.txt"
        requirements = self._freeze_requirements()
        self._atomic_write_text(req_path, requirements)
        
        artifacts = {}
        for artifact_path in [config_path, equity_path, trades_path, 
                              data_manifest_path, req_path]:
            if artifact_path.exists():
                artifacts[artifact_path.name] = {
                    "sha256": self._sha256_of_file(artifact_path),
                    "size_bytes": artifact_path.stat().st_size,
                }
        
        meta = {
            "run_id": run_id,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "git_hash": self._get_git_hash(),
            "python_version": sys.version,
            "artifacts": artifacts,
        }
        
        if extra_meta:
            meta.update(extra_meta)
        
        meta_path = run_dir / "meta.json"
        self._atomic_write_text(meta_path, json.dumps(meta, indent=2, default=str))
        
        # Update manifest
        metrics = {}
        if not equity_df.empty:
            try:
                # heuristic to find equity column
                cand = [c for c in equity_df.columns if "equity" in c.lower() or "nav" in c.lower() or "total_value" in c.lower()]
                col = cand[0] if cand else equity_df.columns[0]
                final_equity = float(equity_df[col].iloc[-1])
                initial_equity = float(equity_df[col].iloc[0])
                total_return = (final_equity / initial_equity) - 1.0
                
                # Sharpe (if returns in meta)
                sharpe = extra_meta.get("sharpe_ratio") if extra_meta else None
                mdd = extra_meta.get("max_drawdown") if extra_meta else None
                
                metrics = {
                    "final_equity": final_equity,
                    "total_return": total_return,
                    "sharpe": sharpe,
                    "max_drawdown": mdd
                }
            except Exception:
                pass
        
        record = {
            "run_id": run_id,
            "timestamp": meta["timestamp_utc"],
            "strategy": config.get("strategy_name", "unknown"),
            "tickers": config.get("tickers", []),
            "git_hash": meta["git_hash"],
            "metrics": metrics,
            "path": str(run_dir),
        }
        
        manifest = self._load_manifest()
        manifest.insert(0, record)
        self._write_manifest(manifest)
        
        return run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        return self._load_manifest()

    def load_run(self, run_id: str) -> Dict[str, Any]:
        manifest = self._load_manifest()
        for r in manifest:
            if r["run_id"] == run_id:
                run_dir = Path(r["path"])
                equity = pd.read_csv(run_dir / "equity.csv", index_col=0, parse_dates=True)
                return {
                    "meta": r,
                    "results": equity,
                }
        raise KeyError(f"Run {run_id} not found.")

