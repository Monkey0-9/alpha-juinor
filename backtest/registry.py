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
        
        Returns:
            dict with 'run_id' and 'path' of the created directory
            
        Raises:
            RuntimeError if directory already exists (immutability guarantee)
        """
        # Generate unique run_id with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        unique_suffix = uuid.uuid4().hex[:8]
        run_id = f"run_{timestamp}_{unique_suffix}"
        
        run_dir = self.base_dir / run_id
        
        # IMMUTABILITY: Never overwrite existing runs
        if run_dir.exists():
            raise RuntimeError(
                f"Run directory {run_id} already exists. "
                "This violates immutability - runs cannot be overwritten."
            )
        
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
        
        Required artifacts:
        - config.json
        - equity.csv
        - trades.csv (can be empty)
        - data_manifest.json
        - meta.json (with git_hash, checksums, python_version)
        - requirements.txt
        
        Args:
            run_id: Run identifier (from create_run)
            config: Strategy configuration dict
            equity_df: Equity curve DataFrame
            trades_df: Trades DataFrame (optional)
            data_manifest: Data source metadata (optional)
            extra_meta: Additional metadata to include (optional)
            
        Returns:
            run_id for reference
        """
        run_dir = self.base_dir / run_id
        
        if not run_dir.exists():
            raise ValueError(f"Run directory {run_id} does not exist. Call create_run first.")
        
        # -------------------------
        # 1. Save config.json
        # -------------------------
        config_path = run_dir / "config.json"
        self._atomic_write_text(config_path, json.dumps(config, indent=2, default=str))
        
        # -------------------------
        # 2. Save equity.csv
        # -------------------------
        equity_path = run_dir / "equity.csv"
        self._atomic_write_df(equity_path, equity_df)
        
        # -------------------------
        # 3. Save trades.csv
        # -------------------------
        trades_path = run_dir / "trades.csv"
        if trades_df is not None and not trades_df.empty:
            self._atomic_write_df(trades_path, trades_df)
        else:
            # Create empty placeholder
            pd.DataFrame().to_csv(trades_path, index=False)
        
        # -------------------------
        # 4. Save data_manifest.json
        # -------------------------
        data_manifest_path = run_dir / "data_manifest.json"
        manifest_data = data_manifest or {}
        self._atomic_write_text(
            data_manifest_path,
            json.dumps(manifest_data, indent=2, default=str)
        )
        
        # -------------------------
        # 5. Save requirements.txt
        # -------------------------
        req_path = run_dir / "requirements.txt"
        requirements = self._freeze_requirements()
        self._atomic_write_text(req_path, requirements)
        
        # -------------------------
        # 6. Compute checksums for all artifacts
        # -------------------------
        artifacts = {}
        for artifact_path in [config_path, equity_path, trades_path, 
                              data_manifest_path, req_path]:
            if artifact_path.exists():
                artifacts[artifact_path.name] = {
                    "sha256": self._sha256_of_file(artifact_path),
                    "size_bytes": artifact_path.stat().st_size,
                }
        
        # -------------------------
        # 7. Save meta.json with git hash, checksums, Python version
        # -------------------------
        meta = {
            "run_id": run_id,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "git_hash": self._get_git_hash(),
            "python_version": sys.version,
            "artifacts": artifacts,
        }
        
        # Add any extra metadata
        if extra_meta:
            meta.update(extra_meta)
        
        meta_path = run_dir / "meta.json"
        self._atomic_write_text(meta_path, json.dumps(meta, indent=2, default=str))
        
        # -------------------------
        # 8. Update central manifest
        # -------------------------
        # Extract metrics from equity for manifest summary
        metrics = {}
        if not equity_df.empty:
            try:
                if 'equity' in equity_df.columns:
                    final_equity = float(equity_df['equity'].iloc[-1])
                    initial_equity = float(equity_df['equity'].iloc[0])
                elif 'total_value' in equity_df.columns:
                    final_equity = float(equity_df['total_value'].iloc[-1])
                    initial_equity = float(equity_df['total_value'].iloc[0])
                else:
                    # Use first numeric column
                    numeric_cols = equity_df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        final_equity = float(equity_df[numeric_cols[0]].iloc[-1])
                        initial_equity = float(equity_df[numeric_cols[0]].iloc[0])
                    else:
                        final_equity = None
                        initial_equity = None
                
                if final_equity and initial_equity:
                    total_return = (final_equity / initial_equity) - 1.0
                    metrics = {
                        "final_equity": final_equity,
                        "total_return": total_return,
                    }
            except Exception:
                pass
        
        record = {
            "run_id": run_id,
            "timestamp": meta["timestamp_utc"],
            "strategy": config.get("strategy", config.get("strategy_name", "unknown")),
            "tickers": config.get("tickers", []),
            "git_hash": meta["git_hash"],
            "metrics": metrics,
            "path": str(run_dir),
        }
        
        manifest = self._load_manifest()
        manifest.insert(0, record)  # newest first
        self._write_manifest(manifest)
        
        return run_id

    def validate_run(self, run_id: str) -> Dict[str, Any]:
        """
        Validate integrity of run artifacts by checking checksums.
        
        Returns:
            dict with validation results
        """
        run_dir = self.base_dir / run_id
        
        if not run_dir.exists():
            return {"valid": False, "error": f"Run {run_id} not found"}
        
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            return {"valid": False, "error": "meta.json not found"}
        
        # Load meta.json
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        stored_checksums = meta.get("artifacts", {})
        
        # Verify each artifact
        mismatches = []
        for artifact_name, stored_info in stored_checksums.items():
            artifact_path = run_dir / artifact_name
            
            if not artifact_path.exists():
                mismatches.append(f"{artifact_name}: file missing")
                continue
            
            # Recompute checksum
            current_checksum = self._sha256_of_file(artifact_path)
            stored_checksum = stored_info.get("sha256")
            
            if current_checksum != stored_checksum:
                mismatches.append(
                    f"{artifact_name}: checksum mismatch "
                    f"(expected {stored_checksum[:8]}..., got {current_checksum[:8]}...)"
                )
        
        return {
            "valid": len(mismatches) == 0,
            "mismatches": mismatches,
            "run_id": run_id,
            "git_hash": meta.get("git_hash"),
        }

    def register_run(
        self,
        config: Dict[str, Any],
        results_df: pd.DataFrame,
        metrics: Dict[str, Any],
        extra_artifacts: Dict[str, pd.DataFrame] | None = None,
    ) -> str:
        """
        Legacy method for backward compatibility.
        
        DEPRECATED: Use create_run() + save_artifacts() instead.
        """
        run_id = uuid.uuid4().hex[:10]
        timestamp = datetime.utcnow().isoformat() + "Z"

        run_dir = self.base_dir / run_id
        run_dir.mkdir(exist_ok=False)

        # Save config
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

        # Save primary result
        results_df.to_csv(run_dir / "equity.csv")

        # Save extra artifacts (optional)
        if extra_artifacts:
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir()
            for name, df in extra_artifacts.items():
                df.to_csv(artifacts_dir / f"{name}.csv")

        # Update manifest
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
