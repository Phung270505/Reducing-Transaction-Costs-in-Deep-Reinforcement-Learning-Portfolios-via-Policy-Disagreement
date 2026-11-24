"""Stage 1: train and save exactly five independent PPO checkpoints."""

from __future__ import annotations

import argparse
import sys

from main import build_parser, configs_from_train_args
from vn30ppo.config import EnsembleConfig
from vn30ppo.experiment import run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    values = sys.argv[1:] if argv is None else argv
    return build_parser().parse_args(["train", *values])


def main() -> None:
    args = parse_args()
    experiment, market, ppo = configs_from_train_args(args)
    run_experiment(
        experiment,
        market,
        ppo,
        seeds=args.seeds,
        ensemble=EnsembleConfig(),
        evaluation_interval=args.evaluation_interval,
        train_only=True,
    )


if __name__ == "__main__":
    main()
