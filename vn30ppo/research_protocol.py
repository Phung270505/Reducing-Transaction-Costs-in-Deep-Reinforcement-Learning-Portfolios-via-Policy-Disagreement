"""Canonical names and invariants for the proposed research protocol.

Keeping these values in one module prevents the training, checkpoint
re-evaluation and walk-forward pipelines from silently treating different
ensemble variants as the paper's proposed strategy.
"""

from __future__ import annotations

from collections.abc import Sequence


COMBINED_ADAPTIVE_STRATEGY = "PPO_robust_asset_adaptive_optimizer"
COMBINED_ADAPTIVE_V1_STRATEGY = "PPO_robust_asset_adaptive_optimizer_frozen_v1"
SOFT_DISAGREEMENT_STRATEGY = "PPO_soft_disagreement_rebalance"
CONSENSUS_ONLY_STRATEGY = "PPO_consensus_only_gate"
PROPOSED_STRATEGY = COMBINED_ADAPTIVE_V1_STRATEGY
FINAL_OPTIMIZED_STRATEGIES = (
    COMBINED_ADAPTIVE_V1_STRATEGY,
    COMBINED_ADAPTIVE_STRATEGY,
    SOFT_DISAGREEMENT_STRATEGY,
)
FINAL_BASELINE_STRATEGIES = (
    "PPO_mean_ensemble",
    CONSENSUS_ONLY_STRATEGY,
    "EqualWeight_monthly",
    "VNINDEX",
)
DEFAULT_RESEARCH_SEEDS = (5, 10, 15, 20, 25)
REQUIRED_ENSEMBLE_SIZE = 5


def validate_research_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate the five independent PPO runs required by the study."""

    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) != REQUIRED_ENSEMBLE_SIZE:
        raise ValueError(
            "The confirmatory disagreement experiment requires exactly "
            f"{REQUIRED_ENSEMBLE_SIZE} independent PPO seeds"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Seeds must be distinct independent training runs")
    if any(seed < 0 for seed in normalized):
        raise ValueError("Seeds must be non-negative")
    return normalized
