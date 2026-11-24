from __future__ import annotations

import pytest

from vn30ppo.research_protocol import (
    COMBINED_ADAPTIVE_STRATEGY,
    COMBINED_ADAPTIVE_V1_STRATEGY,
    CONSENSUS_ONLY_STRATEGY,
    DEFAULT_RESEARCH_SEEDS,
    FINAL_OPTIMIZED_STRATEGIES,
    PROPOSED_STRATEGY,
    SOFT_DISAGREEMENT_STRATEGY,
    validate_research_seeds,
)


def test_confirmatory_protocol_is_one_named_five_seed_strategy() -> None:
    assert PROPOSED_STRATEGY == COMBINED_ADAPTIVE_V1_STRATEGY
    assert FINAL_OPTIMIZED_STRATEGIES == (
        COMBINED_ADAPTIVE_V1_STRATEGY,
        COMBINED_ADAPTIVE_STRATEGY,
        SOFT_DISAGREEMENT_STRATEGY,
    )
    assert CONSENSUS_ONLY_STRATEGY == "PPO_consensus_only_gate"
    assert validate_research_seeds(DEFAULT_RESEARCH_SEEDS) == (5, 10, 15, 20, 25)


@pytest.mark.parametrize("seeds", [[5, 10], [5, 10, 15, 20, 20]])
def test_confirmatory_protocol_rejects_invalid_ensembles(seeds: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_research_seeds(seeds)
