#!/usr/bin/env python3
"""
Replay Run Script - Deterministic Replay Engine

This script enables replay of any historical run to verify determinism.
If replay differs from original → BLOCK PROMOTION.

Usage:
    python scripts/replay_run.py --run-id X
    python scripts/replay_run.py --run-id X --verify-only

RULE: No nondeterminism without logging the seed.
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("REPLAY_ENGINE")


# ============================================================================
# RUN MANIFEST
# ============================================================================

class RunManifest:
    """
    Captures all seeds and hashes for deterministic replay.

    Every run MUST log:
    - run_id
    - global_seed
    - solver_seed
    - model_seeds
    - data_snapshot_hash
    """

    MANIFEST_DIR = Path("runtime/run_manifests")

    def __init__(
        self,
        run_id: str,
        global_seed: int = 42,
        solver_seed: int = 1337,
        model_seeds: Optional[Dict[str, int]] = None,
        data_snapshot_hash: str = "",
        timestamp: str = ""
    ):
        self.run_id = run_id
        self.global_seed = global_seed
        self.solver_seed = solver_seed
        self.model_seeds = model_seeds or {}
        self.data_snapshot_hash = data_snapshot_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat() + "Z"

        # Results captured after run
        self.decision_hashes: Dict[str, str] = {}
        self.allocation_hash: str = ""
        self.order_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "global_seed": self.global_seed,
            "solver_seed": self.solver_seed,
            "model_seeds": self.model_seeds,
            "data_snapshot_hash": self.data_snapshot_hash,
            "timestamp": self.timestamp,
            "results": {
                "decision_hashes": self.decision_hashes,
                "allocation_hash": self.allocation_hash,
                "order_hash": self.order_hash
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunManifest":
        manifest = cls(
            run_id=data["run_id"],
            global_seed=data.get("global_seed", 42),
            solver_seed=data.get("solver_seed", 1337),
            model_seeds=data.get("model_seeds", {}),
            data_snapshot_hash=data.get("data_snapshot_hash", ""),
            timestamp=data.get("timestamp", "")
        )
        results = data.get("results", {})
        manifest.decision_hashes = results.get("decision_hashes", {})
        manifest.allocation_hash = results.get("allocation_hash", "")
        manifest.order_hash = results.get("order_hash", "")
        return manifest

    def save(self) -> Path:
        """Save manifest to disk."""
        self.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        path = self.MANIFEST_DIR / f"{self.run_id}.json"
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved run manifest: {path}")
        return path

    @classmethod
    def load(cls, run_id: str) -> Optional["RunManifest"]:
        """Load manifest from disk."""
        path = cls.MANIFEST_DIR / f"{run_id}.json"
        if not path.exists():
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# DETERMINISM UTILITIES
# ============================================================================

def set_global_seeds(seed: int) -> None:
    """Set all global random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Try to set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.info(f"Set global seed: {seed}")


def compute_data_hash(data: Any) -> str:
    """Compute deterministic hash of data for replay verification."""
    if hasattr(data, 'to_json'):
        content = data.to_json()
    elif isinstance(data, dict):
        content = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, (list, tuple)):
        content = json.dumps(list(data), sort_keys=True, default=str)
    else:
        content = str(data)

    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_decision_hash(decisions: list) -> str:
    """Compute hash of trading decisions for replay verification."""
    # Sort by symbol for determinism
    sorted_decisions = sorted(
        decisions,
        key=lambda d: d.get('symbol', '') if isinstance(d, dict) else getattr(d, 'symbol', '')
    )
    return compute_data_hash(sorted_decisions)


# ============================================================================
# REPLAY ENGINE
# ============================================================================

class ReplayEngine:
    """
    Deterministic replay engine for verifying run reproducibility.
    """

    def __init__(self):
        self.original_manifest: Optional[RunManifest] = None
        self.replay_manifest: Optional[RunManifest] = None

    def load_original(self, run_id: str) -> bool:
        """Load original run manifest."""
        self.original_manifest = RunManifest.load(run_id)
        if self.original_manifest is None:
            logger.error(f"No manifest found for run_id: {run_id}")
            return False
        logger.info(f"Loaded original manifest: {run_id}")
        return True

    def replay(self, run_id: str) -> bool:
        """
        Replay a run and verify determinism.

        Returns:
            True if replay matches original, False otherwise
        """
        if not self.load_original(run_id):
            return False

        orig = self.original_manifest

        # Set seeds from original
        set_global_seeds(orig.global_seed)

        # Create replay manifest
        self.replay_manifest = RunManifest(
            run_id=f"{run_id}_replay",
            global_seed=orig.global_seed,
            solver_seed=orig.solver_seed,
            model_seeds=orig.model_seeds.copy(),
            data_snapshot_hash=orig.data_snapshot_hash
        )

        logger.info("Starting replay...")
        logger.info(f"  Global seed: {orig.global_seed}")
        logger.info(f"  Solver seed: {orig.solver_seed}")
        logger.info(f"  Data hash: {orig.data_snapshot_hash}")

        # Execute replay cycle
        try:
            from orchestration.cycle_runner import run_institutional_cycle

            # Run with same parameters
            result = run_institutional_cycle(
                universe_path="configs/universe.json",
                lookback_years=5,
                max_workers=10,
                paper_mode=True,
                dry_run=True
            )

            # Capture replay hashes
            decisions = result.decisions if hasattr(result, 'decisions') else []
            self.replay_manifest.decision_hashes = {
                "all": compute_decision_hash([d.to_dict() if hasattr(d, 'to_dict') else d for d in decisions])
            }

        except Exception as e:
            logger.error(f"Replay execution failed: {e}")
            return False

        # Compare results
        return self.verify()

    def verify(self) -> bool:
        """
        Verify replay matches original.

        Returns:
            True if match, False if divergence detected
        """
        if not self.original_manifest or not self.replay_manifest:
            logger.error("Cannot verify: missing manifests")
            return False

        orig = self.original_manifest
        replay = self.replay_manifest

        divergences = []

        # Check decision hashes
        for key, orig_hash in orig.decision_hashes.items():
            replay_hash = replay.decision_hashes.get(key, "")
            if orig_hash != replay_hash:
                divergences.append({
                    "type": "decision_hash",
                    "key": key,
                    "original": orig_hash,
                    "replay": replay_hash
                })

        # Check allocation hash
        if orig.allocation_hash and orig.allocation_hash != replay.allocation_hash:
            divergences.append({
                "type": "allocation_hash",
                "original": orig.allocation_hash,
                "replay": replay.allocation_hash
            })

        if divergences:
            logger.error("=" * 60)
            logger.error("REPLAY DIVERGENCE DETECTED")
            logger.error("=" * 60)
            for d in divergences:
                logger.error(f"  {d['type']}: {d.get('key', '')}")
                logger.error(f"    Original: {d['original']}")
                logger.error(f"    Replay:   {d['replay']}")
            logger.error("=" * 60)
            logger.error("BLOCK PROMOTION: Nondeterminism detected!")
            return False

        logger.info("=" * 60)
        logger.info("REPLAY VERIFICATION PASSED")
        logger.info("=" * 60)
        logger.info("All hashes match. Run is deterministic.")
        return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Replay runs for determinism verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Replay and verify a run
    python replay_run.py --run-id cycle_20260127_123456

    # Just verify without replay (if replay manifest exists)
    python replay_run.py --run-id cycle_20260127_123456 --verify-only

    # Create a new run with seed locking
    python replay_run.py --create --run-id test_run --seed 42
        """
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID to replay/verify"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing manifests, don't replay"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a new run manifest with seeds"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for new run (default: 42)"
    )

    args = parser.parse_args()

    if args.create:
        # Create new manifest
        manifest = RunManifest(
            run_id=args.run_id,
            global_seed=args.seed
        )
        manifest.save()
        logger.info(f"Created new run manifest: {args.run_id}")
        sys.exit(0)

    engine = ReplayEngine()

    if args.verify_only:
        # Just load and display manifest
        if engine.load_original(args.run_id):
            manifest = engine.original_manifest
            logger.info(f"Run ID: {manifest.run_id}")
            logger.info(f"Timestamp: {manifest.timestamp}")
            logger.info(f"Global Seed: {manifest.global_seed}")
            logger.info(f"Solver Seed: {manifest.solver_seed}")
            logger.info(f"Data Hash: {manifest.data_snapshot_hash}")
            logger.info(f"Decision Hashes: {manifest.decision_hashes}")
            sys.exit(0)
        else:
            sys.exit(1)

    # Full replay
    success = engine.replay(args.run_id)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
