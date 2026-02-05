# R2P specific validators
# To be implemented in detail during Phase 4 full rollout.

def validate_data_quality_score(score: float, threshold: float = 0.7) -> bool:
    return score >= threshold

def validate_reproducibility(metadata: dict) -> bool:
    return "seed" in metadata and "config_hash" in metadata
