"""Research Platform - Alpha Discovery & Validation."""

from .research_engine import (
    get_alpha_pipeline,
    get_statistical_tester,
    get_walk_forward_analyzer,
)
from .strategy_validator import get_kill_switch
from .optimizer import get_optimizer

__all__ = [
    "get_alpha_pipeline",
    "get_statistical_tester",
    "get_walk_forward_analyzer",
    "get_kill_switch",
    "get_optimizer",
]
