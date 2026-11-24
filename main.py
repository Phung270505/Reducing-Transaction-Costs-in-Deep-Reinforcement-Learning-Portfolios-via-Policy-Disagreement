from __future__ import annotations

import argparse
from dataclasses import replace

from vn30ppo.config import EnsembleConfig, ExperimentConfig, MarketConfig, PPOConfig
from vn30ppo.data import load_market_data
from vn30ppo.experiment import reevaluate_ensemble_from_checkpoints, run_experiment
from vn30ppo.walk_forward import (
    reevaluate_walk_forward_from_checkpoints,
    run_walk_forward_experiment,
)
from vn30ppo.research_protocol import (
    DEFAULT_RESEARCH_SEEDS,
    validate_research_seeds,
)


DEFAULT_TRAIN_OUTPUT = "artifacts/disagreement_ppo_main"
DEFAULT_WALK_FORWARD_OUTPUT = "artifacts/walk_forward_disagreement_ppo"
DEFAULT_UNIVERSE = "Data/VN30_VNINDEX_2016_2025.csv"


def _seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from error
    try:
        return list(validate_research_seeds(seeds))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _year_list(value: str) -> list[int]:
    try:
        years = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Years must be comma-separated integers"
        ) from error
    if not years or len(set(years)) != len(years):
        raise argparse.ArgumentTypeError("Years must be non-empty and distinct")
    if any(year < 1900 or year > 2200 for year in years):
        raise argparse.ArgumentTypeError("Years must be four-digit calendar years")
    return years


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a T+2-aware PPO portfolio on VN30."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="Validate the dataset")
    inspect_parser.add_argument(
        "--data", default="Data/VN30_VNINDEX_2016_2025.csv"
    )
    inspect_parser.add_argument(
        "--universe",
        default=DEFAULT_UNIVERSE,
        help="CSV containing the 30 current VN30 tickers (defaults to the price data)",
    )

    ensemble_parser = subparsers.add_parser(
        "evaluate-ensemble",
        help=(
            "Calibrate disagreement on the disjoint calibration split and reuse "
            "compatible checkpoints"
        ),
    )
    ensemble_parser.add_argument("--artifacts", default=DEFAULT_TRAIN_OUTPUT)
    ensemble_parser.add_argument(
        "--device",
        default="auto",
        help="auto prioritizes cuda:0; cpu forces CPU",
    )

    walk_evaluate_parser = subparsers.add_parser(
        "evaluate-walk-forward",
        help="Apply the current ensemble method to completed annual checkpoints",
    )
    walk_evaluate_parser.add_argument(
        "--artifacts", default=DEFAULT_WALK_FORWARD_OUTPUT
    )
    walk_evaluate_parser.add_argument("--device", default="auto")

    train_parser = subparsers.add_parser("train", help="Run a multi-seed experiment")
    train_parser.add_argument(
        "--data", default="Data/VN30_VNINDEX_2016_2025.csv"
    )
    train_parser.add_argument(
        "--universe",
        default=DEFAULT_UNIVERSE,
        help="CSV containing the 30 current VN30 tickers (defaults to the price data)",
    )
    train_parser.add_argument("--output", default=DEFAULT_TRAIN_OUTPUT)
    train_parser.add_argument(
        "--seeds", type=_seed_list, default=list(DEFAULT_RESEARCH_SEEDS)
    )
    train_parser.add_argument("--timesteps", type=int, default=200_000)
    train_parser.add_argument("--rollout-steps", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--update-epochs", type=int, default=6)
    train_parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    train_parser.add_argument(
        "--architecture", choices=["legacy", "attention"], default="attention"
    )
    train_parser.add_argument("--attention-heads", type=int, default=4)
    train_parser.add_argument("--attention-layers", type=int, default=2)
    train_parser.add_argument("--entropy-coef", type=float, default=0.0005)
    train_parser.add_argument("--final-entropy-coef", type=float, default=0.0)
    train_parser.add_argument("--momentum-prior-topk", type=int, default=15)
    train_parser.add_argument("--momentum-prior-cash-weight", type=float, default=0.05)
    train_parser.add_argument("--momentum-prior-strength", type=float, default=0.25)
    train_parser.add_argument("--penalty-warmup-fraction", type=float, default=0.25)
    train_parser.add_argument("--rebalance-interval", type=int, default=20)
    train_parser.add_argument(
        "--validation-objective",
        choices=["risk_adjusted", "profit"],
        default="profit",
    )
    train_parser.add_argument(
        "--validation-turnover-penalty", type=float, default=0.005
    )
    train_parser.add_argument(
        "--validation-drawdown-penalty", type=float, default=0.10
    )
    train_parser.add_argument(
        "--device",
        default="auto",
        help="auto prioritizes cuda:0 when available; cpu forces CPU",
    )
    train_parser.add_argument("--evaluation-interval", type=int, default=2)
    train_parser.add_argument("--fee-rate", type=float, default=0.0015)
    train_parser.add_argument("--sell-tax-rate", type=float, default=0.001)
    train_parser.add_argument("--slippage-bps", type=float, default=5.0)
    train_parser.add_argument("--settlement-days", type=int, default=2)
    train_parser.add_argument("--turnover-penalty-rate", type=float, default=0.0005)
    train_parser.add_argument("--drawdown-penalty", type=float, default=0.01)
    train_parser.add_argument("--minimum-trade-fraction", type=float, default=0.005)
    train_parser.add_argument("--allocation-alpha-reward", type=float, default=1.0)

    walk_parser = subparsers.add_parser(
        "walk-forward",
        parents=[],
        help="Run expanding-window validation and annual out-of-sample tests",
    )
    # Keep train and walk-forward execution assumptions identical.  All current
    # train arguments are simple value-taking options, so copying their public
    # argparse attributes avoids two drifting CLI definitions.
    for action in train_parser._actions:
        if action.dest == "help" or not action.option_strings:
            continue
        walk_parser.add_argument(
            *action.option_strings,
            default=action.default,
            type=action.type,
            choices=action.choices,
            help=action.help,
            metavar=action.metavar,
        )
    walk_parser.set_defaults(output=DEFAULT_WALK_FORWARD_OUTPUT)
    walk_parser.add_argument(
        "--test-years",
        type=_year_list,
        default=[2021, 2022, 2023, 2024, 2025],
    )
    walk_parser.add_argument(
        "--validation-years",
        type=int,
        default=2,
        help="At least 2: checkpoint validation then disjoint ensemble calibration",
    )
    walk_parser.add_argument(
        "--force",
        action="store_true",
        help="retrain folds even if their complete artifacts already exist",
    )
    return parser


def inspect_data(path: str, universe_path: str) -> None:
    experiment = replace(
        ExperimentConfig(), data_path=path, universe_path=universe_path
    )
    data = load_market_data(
        path,
        train_start=experiment.train_start,
        train_end=experiment.train_end,
        lookback=experiment.lookback,
        minimum_training_days=experiment.minimum_training_days,
        membership_path=experiment.universe_path,
        require_point_in_time_universe=experiment.require_point_in_time_universe,
        required_end=experiment.test_end,
        expected_universe_size=experiment.expected_universe_size,
    )
    eligible = [
        ticker for ticker, flag in zip(data.tickers, data.eligible_assets) if flag
    ]
    excluded = [
        ticker for ticker, flag in zip(data.tickers, data.eligible_assets) if not flag
    ]
    print(f"Trading dates : {data.dates.min().date()} .. {data.dates.max().date()}")
    print(f"Stock tickers : {data.n_assets}")
    print(f"Seen in train : {len(eligible)} - {', '.join(eligible)}")
    print(f"Post-train    : {len(excluded)} - {', '.join(excluded) or 'none'}")
    print(f"Feature shape : {data.stock_features.shape}")
    print(
        "VN30 members  : "
        f"{int(data.universe_mask[0].sum())}..{int(data.universe_mask[-1].sum())} "
        "(static current-universe mask loaded)"
    )


def configs_from_train_args(
    args: argparse.Namespace,
) -> tuple[ExperimentConfig, MarketConfig, PPOConfig]:
    """Build the shared training configuration for CLI entry points."""

    experiment = replace(
        ExperimentConfig(),
        data_path=args.data,
        universe_path=args.universe,
        output_dir=args.output,
    )
    market = replace(
        MarketConfig(),
        settlement_days=args.settlement_days,
        broker_fee_rate=args.fee_rate,
        sell_tax_rate=args.sell_tax_rate,
        slippage_bps=args.slippage_bps,
        turnover_penalty_rate=args.turnover_penalty_rate,
        drawdown_penalty=args.drawdown_penalty,
        minimum_trade_fraction=args.minimum_trade_fraction,
        allocation_alpha_reward=args.allocation_alpha_reward,
    )
    ppo = replace(
        PPOConfig(),
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        architecture=args.architecture,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        entropy_coef=args.entropy_coef,
        final_entropy_coef=args.final_entropy_coef,
        momentum_prior_topk=args.momentum_prior_topk,
        momentum_prior_cash_weight=args.momentum_prior_cash_weight,
        momentum_prior_strength=args.momentum_prior_strength,
        penalty_warmup_fraction=args.penalty_warmup_fraction,
        rebalance_interval=args.rebalance_interval,
        validation_objective=args.validation_objective,
        validation_turnover_penalty=args.validation_turnover_penalty,
        validation_drawdown_penalty=args.validation_drawdown_penalty,
        device=args.device,
    )
    return experiment, market, ppo


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect-data":
        inspect_data(args.data, args.universe)
        return
    if args.command == "evaluate-ensemble":
        reevaluate_ensemble_from_checkpoints(
            args.artifacts, device=args.device
        )
        return
    if args.command == "evaluate-walk-forward":
        reevaluate_walk_forward_from_checkpoints(
            args.artifacts, device=args.device
        )
        return

    experiment, market, ppo = configs_from_train_args(args)
    if args.command == "walk-forward":
        run_walk_forward_experiment(
            experiment,
            market,
            ppo,
            seeds=args.seeds,
            test_years=args.test_years,
            validation_years=args.validation_years,
            ensemble=EnsembleConfig(),
            evaluation_interval=args.evaluation_interval,
            reuse_completed=not args.force,
        )
    else:
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
