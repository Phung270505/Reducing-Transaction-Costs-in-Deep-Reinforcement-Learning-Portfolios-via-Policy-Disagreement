from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import (
    EnsembleConfig,
    ExperimentConfig,
    MarketConfig,
    PPOConfig,
    dataclass_dict,
)
from .evaluation import (
    metrics_table,
    paired_circular_block_bootstrap,
    paired_hac_mean_test,
    performance_metrics,
    save_equity_plot,
)
from .experiment import (
    BACKTEST_PROTOCOL_VERSION,
    file_sha256,
    reevaluate_ensemble_from_checkpoints,
    run_experiment,
)
from .ppo import MODEL_VERSION, TRAINING_ALGORITHM_VERSION, format_duration
from .research_protocol import (
    COMBINED_ADAPTIVE_STRATEGY,
    COMBINED_ADAPTIVE_V1_STRATEGY,
    CONSENSUS_ONLY_STRATEGY,
    PROPOSED_STRATEGY,
    SOFT_DISAGREEMENT_STRATEGY,
    validate_research_seeds,
)


DEFAULT_BASELINES = (
    COMBINED_ADAPTIVE_STRATEGY,
    COMBINED_ADAPTIVE_V1_STRATEGY,
    SOFT_DISAGREEMENT_STRATEGY,
    CONSENSUS_ONLY_STRATEGY,
    "PPO_mean_ensemble",
    "EqualWeight_monthly",
)


def expanding_walk_forward_folds(
    experiment: ExperimentConfig,
    test_years: Sequence[int],
    *,
    validation_years: int = 2,
    subsequent_fold_cash_delay_days: int = 0,
) -> list[ExperimentConfig]:
    """Create nested folds: train -> checkpoint validation -> calibration -> test."""

    if validation_years < 2:
        raise ValueError(
            "validation_years must be at least 2 so checkpoint selection and "
            "ensemble calibration use disjoint periods"
        )
    if subsequent_fold_cash_delay_days < 0:
        raise ValueError("subsequent_fold_cash_delay_days cannot be negative")
    requested_years = [int(year) for year in test_years]
    years = sorted(set(requested_years))
    if not years:
        raise ValueError("at least one walk-forward test year is required")
    if len(years) != len(requested_years):
        raise ValueError("walk-forward test years must be unique")
    if any(right != left + 1 for left, right in zip(years, years[1:])):
        raise ValueError(
            "walk-forward test years must be consecutive; otherwise omitted "
            "market years would be silently treated as zero-return strategy days"
        )
    train_start = pd.Timestamp(experiment.train_start)
    folds: list[ExperimentConfig] = []
    for fold_index, test_year in enumerate(years):
        validation_start_year = test_year - validation_years
        train_end_year = validation_start_year - 1
        train_end = pd.Timestamp(f"{train_end_year}-12-31")
        if train_end <= train_start:
            raise ValueError(
                f"test year {test_year} leaves no valid expanding train window"
            )
        folds.append(
            replace(
                experiment,
                train_end=train_end.strftime("%Y-%m-%d"),
                validation_start=f"{validation_start_year}-01-01",
                validation_end=f"{test_year - 2}-12-31",
                calibration_start=f"{test_year - 1}-01-01",
                calibration_end=f"{test_year - 1}-12-31",
                test_start=f"{test_year}-01-01",
                test_end=f"{test_year}-12-31",
                test_initial_cash_delay_days=(
                    0 if fold_index == 0 else subsequent_fold_cash_delay_days
                ),
            )
        )
    return folds


def _chain_daily_frames(
    frames: Sequence[pd.DataFrame],
    *,
    initial_cash: float,
    broker_fee_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Chain annual folds with explicit boundary liquidation friction.

    Subsequent fold environments must start with settlement-delayed cash.  The
    final observation of each preceding fold is liquidated at its marked value;
    broker fee, sell tax and one-way slippage are deducted before compounding.
    """

    if not frames:
        raise ValueError("cannot chain an empty list of daily frames")
    monetary_columns = {
        "nav",
        "pretrade_nav",
        "available_cash",
        "pending_sale_cash",
        "pending_buy_value",
        "buy_notional",
        "sell_notional",
        "broker_fees",
        "sell_tax",
        "slippage_cost",
        "boundary_liquidation_cost",
    }
    chained: list[pd.DataFrame] = []
    capital = float(initial_cash)
    previous_market_close: float | None = None
    liquidation_rate = (
        float(broker_fee_rate) + float(sell_tax_rate) + float(slippage_bps) / 10_000.0
    )
    if liquidation_rate < 0:
        raise ValueError("boundary liquidation rates cannot be negative")
    for fold_number, source in enumerate(frames, start=1):
        frame = source.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if frame.empty:
            raise ValueError("walk-forward fold contains no daily observations")
        if "market_close" not in frame.columns:
            raise ValueError(
                "market_close is required to chain VNINDEX continuously across folds"
            )
        market_closes = frame["market_close"].to_numpy(dtype=np.float64)
        if not np.isfinite(market_closes).all() or (market_closes <= 0).any():
            raise ValueError("walk-forward fold contains invalid market_close values")
        if previous_market_close is not None:
            first = frame.index[0]
            frame.at[first, "benchmark_return"] = (
                float(frame.at[first, "market_close"]) / previous_market_close - 1.0
            )
        fold_initial = float(frame["pretrade_nav"].iloc[0])
        if fold_initial <= 0:
            raise ValueError("fold initial NAV must be positive")
        raw_returns = frame["daily_return"].to_numpy(dtype=np.float64)
        raw_nav = frame["nav"].to_numpy(dtype=np.float64)
        expected_raw_nav = fold_initial * np.cumprod(1.0 + raw_returns)
        if (
            not np.isfinite(raw_returns).all()
            or not np.isfinite(raw_nav).all()
            or (raw_returns <= -1.0).any()
            or (raw_nav <= 0.0).any()
            or not np.allclose(
                raw_nav,
                expected_raw_nav,
                rtol=1e-8,
                atol=max(1e-6, fold_initial * 1e-10),
            )
        ):
            raise ValueError(
                "walk-forward fold NAV is inconsistent with its daily returns"
            )
        frame["boundary_liquidation_cost"] = 0.0
        frame["boundary_liquidation"] = False
        if fold_number < len(frames):
            if "weight_CASH" not in frame.columns:
                raise ValueError(
                    "weight_CASH is required to charge boundary liquidation "
                    "costs without guessing stock exposure"
                )
            last = frame.index[-1]
            stock_exposure = float(
                np.clip(1.0 - float(frame.at[last, "weight_CASH"]), 0.0, 1.0)
            )
            nav_before = float(frame.at[last, "nav"])
            sell_notional = nav_before * stock_exposure
            broker_fee = sell_notional * float(broker_fee_rate)
            sell_tax = sell_notional * float(sell_tax_rate)
            slippage_cost = sell_notional * float(slippage_bps) / 10_000.0
            liquidation_cost = broker_fee + sell_tax + slippage_cost
            prior_nav = nav_before / max(
                1.0 + float(frame.at[last, "daily_return"]), 1e-12
            )
            frame.at[last, "daily_return"] = (
                (nav_before - liquidation_cost) / prior_nav - 1.0
            )
            frame.at[last, "boundary_liquidation_cost"] = liquidation_cost
            frame.at[last, "boundary_liquidation"] = True
            if "broker_fees" in frame:
                frame.at[last, "broker_fees"] += broker_fee
            if "sell_tax" in frame:
                frame.at[last, "sell_tax"] += sell_tax
            if "slippage_cost" in frame:
                frame.at[last, "slippage_cost"] += slippage_cost
            if "sell_notional" in frame:
                frame.at[last, "sell_notional"] += sell_notional
            if "turnover" in frame:
                frame.at[last, "turnover"] += stock_exposure
            if "available_cash" in frame:
                frame.at[last, "available_cash"] = 0.0
            if "pending_sale_cash" in frame:
                frame.at[last, "pending_sale_cash"] = nav_before - liquidation_cost
            if "pending_buy_value" in frame:
                frame.at[last, "pending_buy_value"] = 0.0
            stock_weight_columns = [
                column
                for column in frame.columns
                if column.startswith("weight_") and column != "weight_CASH"
            ]
            if stock_weight_columns:
                frame.loc[last, stock_weight_columns] = 0.0
            frame.at[last, "weight_CASH"] = 1.0
        scale = capital / fold_initial
        for column in monetary_columns.intersection(frame.columns):
            frame[column] = frame[column].astype(float) * scale
        # Reconstruct NAV from returns so every boundary compounds exactly.
        frame["nav"] = capital * (1.0 + frame["daily_return"]).cumprod()
        frame["walk_forward_fold"] = fold_number
        capital = float(frame["nav"].iloc[-1])
        previous_market_close = float(frame["market_close"].iloc[-1])
        chained.append(frame)
    result = pd.concat(chained, ignore_index=True)
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("walk-forward test folds must be disjoint and chronological")
    return result


def aggregate_walk_forward_results(
    output_dir: str | Path,
    folds: Sequence[ExperimentConfig],
    *,
    initial_cash: float,
    broker_fee_rate: float = 0.0,
    sell_tax_rate: float = 0.0,
    slippage_bps: float = 0.0,
    strategy_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate fold artifacts, metrics, calibrations and paired inference."""

    root = Path(output_dir)
    if not folds:
        raise ValueError("at least one fold is required")
    requested = tuple(strategy_names or (PROPOSED_STRATEGY, *DEFAULT_BASELINES))
    histories: dict[str, pd.DataFrame] = {}
    aggregate_metrics: dict[str, dict[str, float]] = {}
    fold_metric_frames: list[pd.DataFrame] = []
    selections: list[dict[str, object]] = []

    for strategy in requested:
        parts: list[pd.DataFrame] = []
        for fold in folds:
            test_year = pd.Timestamp(fold.test_start).year
            fold_dir = root / f"fold_{test_year}"
            daily_path = fold_dir / f"daily_{strategy}.csv"
            if not daily_path.exists():
                parts = []
                break
            part = pd.read_csv(daily_path, parse_dates=["date"])
            part = pd.concat(
                [
                    part,
                    pd.Series(
                        test_year,
                        index=part.index,
                        name="walk_forward_test_year",
                    ),
                ],
                axis=1,
            )
            parts.append(part)
        if not parts:
            continue
        chained = _chain_daily_frames(
            parts,
            initial_cash=initial_cash,
            broker_fee_rate=broker_fee_rate,
            sell_tax_rate=sell_tax_rate,
            slippage_bps=slippage_bps,
        )
        histories[strategy] = chained
        aggregate_metrics[strategy] = performance_metrics(
            chained, initial_cash=initial_cash
        )
        chained.to_csv(root / f"daily_walk_forward_{strategy}.csv", index=False)

    if PROPOSED_STRATEGY not in histories:
        raise FileNotFoundError(
            f"all folds must contain daily_{PROPOSED_STRATEGY}.csv"
        )

    for fold in folds:
        test_year = pd.Timestamp(fold.test_start).year
        fold_dir = root / f"fold_{test_year}"
        fold_metrics = pd.read_csv(fold_dir / "metrics.csv", index_col=0)
        fold_metrics.insert(0, "test_year", test_year)
        fold_metrics.insert(0, "strategy", fold_metrics.index)
        fold_metric_frames.append(fold_metrics.reset_index(drop=True))
        calibration_path = fold_dir / "disagreement_calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        selected_policy = calibration.get("selected_policy")
        if selected_policy is None:
            selected_policy = calibration["selected_policies"][
                PROPOSED_STRATEGY
            ]
        selections.append(
            {
                "test_year": test_year,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "calibration_start": fold.calibration_start,
                "calibration_end": fold.calibration_end,
                **selected_policy,
            }
        )

    pd.concat(fold_metric_frames, ignore_index=True).to_csv(
        root / "walk_forward_fold_metrics.csv", index=False
    )
    pd.DataFrame(selections).to_csv(
        root / "walk_forward_selected_policies.csv", index=False
    )
    table = metrics_table(aggregate_metrics)
    table.to_csv(root / "walk_forward_metrics.csv")
    save_equity_plot(
        histories,
        root / "walk_forward_equity_curves.png",
        title="Strict expanding-window walk-forward performance",
    )

    proposed = histories[PROPOSED_STRATEGY]
    inference_rows: list[dict[str, float | str]] = []
    for baseline in (
        "PPO_mean_ensemble",
        "Momentum60_Top15_Monthly",
        "InverseVolatility60_Monthly",
        "EqualWeight_buy_hold",
        "VNINDEX",
    ):
        if baseline == "VNINDEX":
            baseline_returns = proposed["benchmark_return"].to_numpy(float)
        elif baseline in histories:
            baseline_returns = histories[baseline]["daily_return"].to_numpy(float)
        else:
            continue
        inference_rows.append(
            {
                "strategy": PROPOSED_STRATEGY,
                "baseline": baseline,
                **paired_circular_block_bootstrap(
                    proposed["daily_return"].to_numpy(float), baseline_returns
                ),
                **paired_hac_mean_test(
                    proposed["daily_return"].to_numpy(float), baseline_returns
                ),
            }
        )
    pd.DataFrame(inference_rows).to_csv(
        root / "walk_forward_paired_block_bootstrap.csv", index=False
    )

    # Execution efficiency is a first-class outcome under Vietnam's T+2 and
    # explicit fees/tax/slippage. Negating both series makes a positive paired effect
    # mean that the proposed policy uses less turnover or cost than Momentum.
    momentum_name = "Momentum60_Top15_Monthly"
    execution_baselines = (
        momentum_name,
        "PPO_mean_ensemble",
    )
    if any(name in histories for name in execution_baselines):
        proposed_cost_ratio = (
            proposed["broker_fees"].to_numpy(float)
            + proposed["sell_tax"].to_numpy(float)
            + (
                proposed["slippage_cost"].to_numpy(float)
                if "slippage_cost" in proposed
                else 0.0
            )
        ) / proposed["pretrade_nav"].to_numpy(float)
        execution_rows: list[dict[str, float | str]] = []
        for baseline_name in execution_baselines:
            if baseline_name not in histories:
                continue
            baseline = histories[baseline_name]
            baseline_cost_ratio = (
                baseline["broker_fees"].to_numpy(float)
                + baseline["sell_tax"].to_numpy(float)
                + (
                    baseline["slippage_cost"].to_numpy(float)
                    if "slippage_cost" in baseline
                    else 0.0
                )
            ) / baseline["pretrade_nav"].to_numpy(float)
            for metric, proposed_values, baseline_values, units in (
                (
                    "turnover_reduction",
                    -proposed["turnover"].to_numpy(float),
                    -baseline["turnover"].to_numpy(float),
                    "NAV multiples per year",
                ),
                (
                    "transaction_cost_ratio_reduction",
                    -proposed_cost_ratio,
                    -baseline_cost_ratio,
                    "fraction of NAV per year",
                ),
            ):
                execution_rows.append(
                    {
                        "strategy": PROPOSED_STRATEGY,
                        "baseline": baseline_name,
                        "metric": metric,
                        "units": units,
                        **paired_circular_block_bootstrap(
                            proposed_values, baseline_values
                        ),
                        **paired_hac_mean_test(
                            proposed_values, baseline_values
                        ),
                    }
                )
        pd.DataFrame(execution_rows).to_csv(
            root / "walk_forward_execution_efficiency.csv", index=False
        )

    diversity_rows: list[dict[str, float | int | str]] = []
    review = proposed["disagreement_mean_pairwise_tv"].notna()
    for year, group in proposed.loc[review].groupby(
        proposed.loc[review, "walk_forward_test_year"]
    ):
        diversity_rows.append(
            {
                "strategy": PROPOSED_STRATEGY,
                "test_year": int(year),
                "review_days": int(len(group)),
                "mean_pairwise_tv": float(
                    group["disagreement_mean_pairwise_tv"].mean()
                ),
                "maximum_pairwise_tv": float(
                    group["disagreement_max_pairwise_tv"].max()
                ),
                "mean_rebalance_gate": float(group["rebalance_gate"].mean()),
                "mean_consensus_requested_change": float(
                    group["consensus_requested_change"].mean()
                ),
                "mean_adaptive_requested_change": float(
                    group["adaptive_requested_change"].mean()
                ),
            }
        )
    pd.DataFrame(diversity_rows).to_csv(
        root / "walk_forward_diversity_diagnostics.csv", index=False
    )
    return table


def reevaluate_walk_forward_from_checkpoints(
    output_dir: str | Path,
    *,
    device: str = "auto",
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply the current ensemble algorithm to every completed fold."""

    root = Path(output_dir)
    manifest_path = root / "walk_forward_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing walk-forward manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds: list[ExperimentConfig] = []
    for index, year in enumerate(manifest["test_years"], start=1):
        fold_dir = root / f"fold_{int(year)}"
        config_path = fold_dir / "experiment_config.json"
        configuration = json.loads(config_path.read_text(encoding="utf-8"))
        fold = ExperimentConfig(**configuration["experiment"])
        fold.output_dir = str(fold_dir)
        folds.append(fold)
        if verbose:
            print(
                f"\n=== Re-evaluating fold {index}/{len(manifest['test_years'])} "
                f"test={year} ===",
                flush=True,
            )
        reevaluate_ensemble_from_checkpoints(
            fold_dir,
            device=device,
            verbose=verbose,
        )
    first_configuration = json.loads(
        (root / f"fold_{manifest['test_years'][0]}" / "experiment_config.json")
        .read_text(encoding="utf-8")
    )
    initial_cash = float(first_configuration["market"]["initial_cash"])
    broker_fee_rate = float(first_configuration["market"]["broker_fee_rate"])
    sell_tax_rate = float(first_configuration["market"]["sell_tax_rate"])
    slippage_bps = float(first_configuration["market"]["slippage_bps"])
    table = aggregate_walk_forward_results(
        root,
        folds,
        initial_cash=initial_cash,
        broker_fee_rate=broker_fee_rate,
        sell_tax_rate=sell_tax_rate,
        slippage_bps=slippage_bps,
    )
    manifest["proposed_strategy"] = PROPOSED_STRATEGY
    manifest["ensemble_reevaluated_from_existing_checkpoints"] = True
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        columns = [
            "total_return",
            "cagr",
            "sharpe_ratio",
            "maximum_drawdown",
            "annual_turnover",
        ]
        print("\n=== Re-evaluated chained walk-forward metrics ===")
        print(table.reindex(columns=columns).round(4).to_string())
    return table


def run_walk_forward_experiment(
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    *,
    seeds: list[int],
    test_years: Sequence[int],
    validation_years: int = 2,
    ensemble: EnsembleConfig | None = None,
    evaluation_interval: int = 2,
    reuse_completed: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train and evaluate a leakage-safe expanding-window experiment."""

    started = time.perf_counter()
    seeds = list(validate_research_seeds(seeds))
    root = Path(experiment.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    folds = expanding_walk_forward_folds(
        experiment,
        test_years,
        validation_years=validation_years,
        # At an annual boundary, a position purchased near year-end may first
        # need its shares to settle and then its sale proceeds to settle. Using
        # 2*T is conservative for the mixed available/pending portfolio and
        # prevents early reuse of synthetic liquidation cash.
        subsequent_fold_cash_delay_days=2 * market.settlement_days,
    )
    ensemble = ensemble or EnsembleConfig()
    manifest = {
        "protocol": "point_in_time_nested_validation_with_settled_boundaries",
        "test_years": [pd.Timestamp(fold.test_start).year for fold in folds],
        "validation_years": validation_years,
        "seeds": seeds,
        "proposed_strategy": PROPOSED_STRATEGY,
        "selection_uses_test_data_in_code": False,
        "confirmatory_status": "development_only_until_algorithm_and_holdout_are_frozen",
        "boundary_liquidation": {
            "broker_fee_rate": market.broker_fee_rate,
            "sell_tax_rate": market.sell_tax_rate,
            "slippage_bps": market.slippage_bps,
            "next_fold_cash_delay_days": 2 * market.settlement_days,
            "cash_delay_policy": (
                "conservative buy-settlement plus sale-settlement"
            ),
        },
        "folds": [
            {
                "train": [fold.train_start, fold.train_end],
                "validation": [fold.validation_start, fold.validation_end],
                "calibration": [fold.calibration_start, fold.calibration_end],
                "test": [fold.test_start, fold.test_end],
                "test_initial_cash_delay_days": fold.test_initial_cash_delay_days,
            }
            for fold in folds
        ],
    }
    (root / "walk_forward_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for fold_index, fold in enumerate(folds, start=1):
        test_year = pd.Timestamp(fold.test_start).year
        fold_dir = root / f"fold_{test_year}"
        completed = (
            (fold_dir / "metrics.csv").exists()
            and (fold_dir / f"daily_{PROPOSED_STRATEGY}.csv").exists()
            and all((fold_dir / f"ppo_seed_{seed}.pt").exists() for seed in seeds)
        )
        if completed:
            saved_config_path = fold_dir / "experiment_config.json"
            if not saved_config_path.exists():
                completed = False
            else:
                saved_config = json.loads(
                    saved_config_path.read_text(encoding="utf-8")
                )
                expected_experiment = replace(fold, output_dir=str(fold_dir))
                expected_inputs = {
                    "prices_sha256": file_sha256(expected_experiment.data_path),
                    "universe_sha256": file_sha256(
                        expected_experiment.universe_path
                    ),
                }
                checkpoint_files = saved_config.get("checkpoint_files", {})
                checkpoint_hashes_match = isinstance(checkpoint_files, dict) and all(
                    isinstance(checkpoint_files.get(str(seed)), dict)
                    and checkpoint_files[str(seed)].get("filename")
                    == f"ppo_seed_{seed}.pt"
                    and checkpoint_files[str(seed)].get("sha256")
                    == file_sha256(fold_dir / f"ppo_seed_{seed}.pt")
                    for seed in seeds
                )
                completed = all(
                    (
                        saved_config.get("backtest_protocol_version")
                        == BACKTEST_PROTOCOL_VERSION,
                        saved_config.get("model_version") == MODEL_VERSION,
                        saved_config.get("training_algorithm_version")
                        == TRAINING_ALGORITHM_VERSION,
                        saved_config.get("experiment")
                        == dataclass_dict(expected_experiment),
                        saved_config.get("market") == dataclass_dict(market),
                        saved_config.get("ppo") == dataclass_dict(ppo),
                        saved_config.get("ensemble") == dataclass_dict(ensemble),
                        saved_config.get("evaluation_interval")
                        == evaluation_interval,
                        saved_config.get("seeds") == seeds,
                        saved_config.get("input_files") == expected_inputs,
                        checkpoint_hashes_match,
                    )
                )
        if reuse_completed and completed:
            if verbose:
                print(
                    f"\n=== Fold {fold_index}/{len(folds)} test={test_year} "
                    "already complete; reusing artifacts ===",
                    flush=True,
                )
            continue
        if verbose:
            print(
                f"\n=== Walk-forward fold {fold_index}/{len(folds)} ===\n"
                f"train={fold.train_start}..{fold.train_end} | "
                f"validation={fold.validation_start}..{fold.validation_end} | "
                f"calibration={fold.calibration_start}..{fold.calibration_end} | "
                f"test={fold.test_start}..{fold.test_end}",
                flush=True,
            )
        run_experiment(
            replace(fold, output_dir=str(fold_dir)),
            market,
            ppo,
            seeds=seeds,
            ensemble=ensemble,
            evaluation_interval=evaluation_interval,
            verbose=verbose,
        )

    table = aggregate_walk_forward_results(
        root,
        folds,
        initial_cash=market.initial_cash,
        broker_fee_rate=market.broker_fee_rate,
        sell_tax_rate=market.sell_tax_rate,
        slippage_bps=market.slippage_bps,
    )
    elapsed = time.perf_counter() - started
    manifest["elapsed_seconds"] = elapsed
    manifest["elapsed_display"] = format_duration(elapsed)
    (root / "walk_forward_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        columns = [
            "total_return",
            "cagr",
            "sharpe_ratio",
            "maximum_drawdown",
            "annual_turnover",
            "total_transaction_cost",
        ]
        print("\n=== Chained walk-forward out-of-sample metrics ===")
        print(table.reindex(columns=columns).round(4).to_string())
        print(f"\nWalk-forward time: {format_duration(elapsed)}")
        print(f"Saved walk-forward study to: {root.resolve()}")
    return table
