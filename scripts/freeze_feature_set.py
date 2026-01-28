#!/usr/bin/env python3
"""
Feature Freeze Script - Lock feature sets for model versions.

This script creates immutable feature schema definitions that are used
to validate all future model inputs.

Usage:
    python scripts/freeze_feature_set.py --model ml_v1
    python scripts/freeze_feature_set.py --model ml_v1 --from-config configs/feature_schema.json

RULE: If feature hash ≠ live feature store → disable model automatically.
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.input_schema_guard import get_schema_guard, FrozenFeatureSchema

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("FEATURE_FREEZE")


def load_features_from_schema(schema_path: str) -> list:
    """Load feature names from feature schema JSON."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    features = []
    for feature_def in schema.get("features", []):
        if isinstance(feature_def, dict):
            features.append(feature_def.get("name"))
        elif isinstance(feature_def, str):
            features.append(feature_def)

    return [f for f in features if f]


def load_features_from_model_metadata(model_id: str) -> list:
    """Load feature names from model metadata if available."""
    metadata_paths = [
        Path(f"models/{model_id}_metadata.json"),
        Path(f"ml_models/{model_id}_metadata.json"),
        Path(f"runtime/models/{model_id}_metadata.json"),
    ]

    for path in metadata_paths:
        if path.exists():
            with open(path, 'r') as f:
                metadata = json.load(f)
            return metadata.get("feature_names", [])

    return []


def freeze_feature_set(model_id: str, source: str = None) -> FrozenFeatureSchema:
    """
    Freeze the feature set for a model.

    Args:
        model_id: Model identifier (e.g., "ml_v1")
        source: Optional path to feature schema JSON

    Returns:
        FrozenFeatureSchema object
    """
    # Determine feature source
    if source:
        logger.info(f"Loading features from: {source}")
        feature_names = load_features_from_schema(source)
    else:
        # Try model metadata first
        feature_names = load_features_from_model_metadata(model_id)

        if not feature_names:
            # Fall back to default schema
            default_schema = Path("configs/feature_schema.json")
            if default_schema.exists():
                logger.info(f"Loading features from default schema: {default_schema}")
                feature_names = load_features_from_schema(str(default_schema))
            else:
                logger.error("No feature source found!")
                sys.exit(1)

    if not feature_names:
        logger.error("No features found to freeze!")
        sys.exit(1)

    logger.info(f"Found {len(feature_names)} features to freeze")

    # Freeze using the guard
    guard = get_schema_guard()
    schema = guard.freeze_schema(model_id, feature_names)

    logger.info(f"✓ Frozen feature set: {schema.feature_set_id}")
    logger.info(f"  Hash: {schema.feature_hash}")
    logger.info(f"  Features: {len(schema.feature_names)}")
    logger.info(f"  Stored at: configs/frozen_features/{model_id}.json")

    return schema


def verify_frozen_schema(model_id: str) -> bool:
    """Verify that a frozen schema exists and is valid."""
    guard = get_schema_guard()
    schema = guard.get_frozen_schema(model_id)

    if schema is None:
        logger.error(f"No frozen schema found for {model_id}")
        return False

    logger.info(f"✓ Verified frozen schema for {model_id}")
    logger.info(f"  Feature Set ID: {schema.feature_set_id}")
    logger.info(f"  Feature Count: {schema.feature_count}")
    logger.info(f"  Frozen At: {schema.frozen_at}")
    logger.info(f"  Hash: {schema.feature_hash}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Freeze feature sets for ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Freeze from default config
    python freeze_feature_set.py --model ml_v1

    # Freeze from specific schema
    python freeze_feature_set.py --model ml_v1 --from-config configs/feature_schema.json

    # Verify existing freeze
    python freeze_feature_set.py --model ml_v1 --verify
        """
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier (e.g., ml_v1)"
    )
    parser.add_argument(
        "--from-config",
        dest="source",
        help="Path to feature schema JSON"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing frozen schema instead of creating new"
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_frozen_schema(args.model)
        sys.exit(0 if success else 1)
    else:
        freeze_feature_set(args.model, args.source)
        logger.info("Feature freeze complete!")


if __name__ == "__main__":
    main()
