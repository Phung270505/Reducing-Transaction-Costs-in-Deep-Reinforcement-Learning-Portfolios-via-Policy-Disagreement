from __future__ import annotations

import pytest

from main import (
    DEFAULT_TRAIN_OUTPUT,
    DEFAULT_WALK_FORWARD_OUTPUT,
    build_parser,
)
from optimize_ensemble import build_parser as build_optimize_parser
from train_ppo import parse_args as parse_train_ppo_args


def test_research_commands_use_current_outputs_and_five_distinct_seeds() -> None:
    parser = build_parser()
    train = parser.parse_args(["train"])
    walk = parser.parse_args(["walk-forward"])
    evaluate = parser.parse_args(["evaluate-ensemble"])
    evaluate_walk = parser.parse_args(["evaluate-walk-forward"])

    assert train.output == DEFAULT_TRAIN_OUTPUT
    assert train.universe == train.data
    assert train.timesteps == 200_000
    assert train.seeds == [5, 10, 15, 20, 25]
    assert walk.output == DEFAULT_WALK_FORWARD_OUTPUT
    assert walk.universe == walk.data
    assert walk.timesteps == 200_000
    assert walk.seeds == [5, 10, 15, 20, 25]
    assert walk.test_years == [2021, 2022, 2023, 2024, 2025]
    assert evaluate.artifacts == DEFAULT_TRAIN_OUTPUT
    assert evaluate_walk.artifacts == DEFAULT_WALK_FORWARD_OUTPUT


def test_split_stage_entry_points_share_the_main_artifact_contract() -> None:
    train = parse_train_ppo_args([])
    optimize = build_optimize_parser().parse_args([])
    assert train.command == "train"
    assert train.output == DEFAULT_TRAIN_OUTPUT
    assert train.seeds == [5, 10, 15, 20, 25]
    assert optimize.artifacts == DEFAULT_TRAIN_OUTPUT
    assert not hasattr(optimize, "extended_ablations")


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--seeds", "5,5"],
        ["train", "--seeds", "-1,5"],
        ["train", "--seeds", "5,10"],
        ["walk-forward", "--test-years", "2021,2021"],
    ],
)
def test_cli_rejects_invalid_research_identifiers(arguments: list[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arguments)
