"""Stage 2: optimize and evaluate an ensemble from five saved PPOs."""

from __future__ import annotations

import argparse

from main import DEFAULT_TRAIN_OUTPUT
from vn30ppo.experiment import reevaluate_ensemble_from_checkpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load five PPO checkpoints, calibrate the three retained "
            "disagreement strategies and final baselines, then evaluate "
            "without retraining."
        )
    )
    parser.add_argument("--artifacts", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto prioritizes cuda:0; cpu forces CPU",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reevaluate_ensemble_from_checkpoints(
        args.artifacts,
        device=args.device,
    )


if __name__ == "__main__":
    main()
