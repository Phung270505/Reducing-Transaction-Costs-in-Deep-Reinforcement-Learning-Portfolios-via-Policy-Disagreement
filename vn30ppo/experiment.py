from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EnsembleConfig,
    ExperimentConfig,
    MarketConfig,
    PPOConfig,
    dataclass_dict,
)
from .data import MarketDataBundle, load_market_data
from .environment import VietnamStockEnv
from .evaluation import (
    AdaptiveRebalancePolicy,
    DirectionalDisagreementPolicy,
    DisagreementOptimizationPolicy,
    DisagreementPolicy,
    ReliabilityWeightedPolicy,
    RegimeDisagreementPolicy,
    evaluate_disagreement_ensemble,
    evaluate_directional_disagreement_ensemble,
    evaluate_agent,
    evaluate_adaptive_rebalance_ensemble,
    evaluate_cross_sectional_momentum,
    evaluate_equal_weight,
    evaluate_equal_weight_buy_and_hold,
    evaluate_inverse_volatility,
    evaluate_mean_ensemble,
    evaluate_optimized_disagreement_ensemble,
    evaluate_regime_disagreement_ensemble,
    evaluate_reliability_weighted_ensemble,
    metrics_table,
    paired_circular_block_bootstrap,
    paired_hac_mean_test,
    save_equity_plot,
)
from .ppo import (
    MODEL_VERSION,
    TRAINING_ALGORITHM_VERSION,
    PPOAgent,
    format_device_banner,
    format_duration,
    training_device_info,
)
from .research_protocol import (
    COMBINED_ADAPTIVE_STRATEGY,
    COMBINED_ADAPTIVE_V1_STRATEGY,
    CONSENSUS_ONLY_STRATEGY,
    FINAL_OPTIMIZED_STRATEGIES,
    PROPOSED_STRATEGY,
    SOFT_DISAGREEMENT_STRATEGY,
    validate_research_seeds,
)


BACKTEST_PROTOCOL_VERSION = "3.0.0-proposal-aligned-disagreement-gate"


def file_sha256(path: str | Path) -> str:
    """Return a stable content fingerprint for a research input file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_fingerprints(
    configuration: dict[str, object], experiment: ExperimentConfig
) -> None:
    """Refuse evaluation when research inputs differ from training inputs."""

    saved = configuration.get("input_files")
    current = {
        "prices_sha256": file_sha256(experiment.data_path),
        "universe_sha256": file_sha256(experiment.universe_path),
    }
    if saved != current:
        raise ValueError(
            "Price or point-in-time membership data changed after training. "
            "Checkpoint evaluation is not reproducible; retrain instead."
        )


def make_environment(
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    *,
    split: str,
    seed: int = 42,
) -> VietnamStockEnv:
    periods = {
        "train": (experiment.train_start, experiment.train_end),
        "validation": (experiment.validation_start, experiment.validation_end),
        "calibration": (experiment.calibration_start, experiment.calibration_end),
        "test": (experiment.test_start, experiment.test_end),
    }
    if split not in periods:
        raise ValueError(f"Unknown split: {split}")
    start, end = periods[split]
    return VietnamStockEnv(
        data,
        start_date=start,
        end_date=end,
        lookback=experiment.lookback,
        config=replace(market),
        random_start=split == "train",
        episode_days=(experiment.training_episode_days if split == "train" else None),
        initial_cash_delay_days=(
            experiment.test_initial_cash_delay_days if split == "test" else 0
        ),
        seed=seed,
    )


def _vnindex_row(reference_metrics: dict[str, float]) -> dict[str, float]:
    return {
        "total_return": reference_metrics["benchmark_total_return"],
        "cagr": reference_metrics["benchmark_cagr"],
        "sharpe_ratio": reference_metrics["benchmark_sharpe"],
        "maximum_drawdown": reference_metrics["benchmark_maximum_drawdown"],
        "total_turnover": 0.0,
        "annual_turnover": 0.0,
        "trade_count": 0.0,
        "broker_fees": 0.0,
        "sell_tax": 0.0,
        "slippage_cost": 0.0,
        "total_transaction_cost": 0.0,
        "transaction_cost_to_initial": 0.0,
    }


def _save_paired_bootstrap_comparisons(
    histories: dict[str, pd.DataFrame],
    proposed_name: str,
    output_dir: Path,
) -> None:
    if proposed_name not in histories:
        return
    proposed = histories[proposed_name]
    rows: list[dict[str, float | str]] = []
    for baseline_name in (
        "Momentum60_Top15_Monthly",
        "InverseVolatility60_Monthly",
        "EqualWeight_buy_hold",
        "VNINDEX",
    ):
        if baseline_name == "VNINDEX":
            baseline_returns = proposed["benchmark_return"].to_numpy()
        elif baseline_name in histories:
            baseline_returns = histories[baseline_name]["daily_return"].to_numpy()
        else:
            continue
        statistics = paired_circular_block_bootstrap(
            proposed["daily_return"].to_numpy(), baseline_returns
        )
        hac_statistics = paired_hac_mean_test(
            proposed["daily_return"].to_numpy(), baseline_returns
        )
        rows.append(
            {
                "strategy": proposed_name,
                "baseline": baseline_name,
                **statistics,
                **hac_statistics,
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "paired_block_bootstrap.csv", index=False
    )


def _merge_reevaluated_results(
    output_dir: Path,
    results: dict[str, dict[str, float]],
    *,
    proposed_name: str,
) -> pd.DataFrame:
    """Merge new ensemble rows while preserving trained seeds and baselines."""

    metrics_path = output_dir / "metrics.csv"
    table = (
        pd.read_csv(metrics_path, index_col=0)
        if metrics_path.exists()
        else pd.DataFrame()
    )
    legacy_strategy_names = {
        "PPO_disagreement_adaptive",
        "PPO_signed_asset_disagreement_ablation",
    }
    table = table.drop(index=list(legacy_strategy_names), errors="ignore")
    for strategy, values in results.items():
        for metric, value in values.items():
            table.loc[strategy, metric] = value
    table.index.name = "strategy"
    table.to_csv(metrics_path)

    histories: dict[str, pd.DataFrame] = {}
    for daily_path in output_dir.glob("daily_*.csv"):
        name = daily_path.stem.removeprefix("daily_")
        if name in legacy_strategy_names:
            continue
        histories[name] = pd.read_csv(daily_path, parse_dates=["date"])
    _save_paired_bootstrap_comparisons(
        histories, proposed_name, output_dir
    )
    save_equity_plot(histories, output_dir / "equity_curves.png")
    return table


def _ensemble_reliability_objective(
    metrics: dict[str, float], ppo: PPOConfig
) -> float:
    """Risk-adjusted validation score for the equal-seed adaptive gate."""

    sharpe = float(metrics["sharpe_ratio"])
    turnover = float(metrics["annual_turnover"])
    drawdown = abs(float(metrics["maximum_drawdown"]))
    if not all(np.isfinite(value) for value in (sharpe, turnover, drawdown)):
        return -np.inf
    return float(
        sharpe
        - ppo.validation_turnover_penalty * turnover
        - ppo.validation_drawdown_penalty * drawdown
    )


def _temporal_validation_score(daily: pd.DataFrame) -> tuple[float, float]:
    """Penalize unstable calibration subperiods without using test data.

    Main and walk-forward calibration windows are one calendar year. Scoring
    only annual returns would therefore have zero dispersion and would reduce
    model selection to in-sample total return. We use calendar quarters when
    fewer than two complete years are available, and years otherwise.
    """

    if daily.empty:
        raise ValueError("validation backtest contains no returns")
    frame = daily[["date", "daily_return"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    years = frame["date"].dt.year.nunique()
    frame["validation_period"] = (
        frame["date"].dt.to_period("Y")
        if years >= 2
        else frame["date"].dt.to_period("Q")
    )
    period_returns = (
        frame.groupby("validation_period")["daily_return"]
        .apply(lambda values: float(np.prod(1.0 + values) - 1.0))
        .to_numpy(dtype=np.float64)
    )
    if period_returns.size < 2:
        raise ValueError(
            "calibration requires at least two temporal subperiods"
        )
    score = float(period_returns.mean() - 0.5 * period_returns.std(ddof=0))
    return score, float(period_returns.min())


def disagreement_quantile_report(
    daily: pd.DataFrame, *, groups: int = 5
) -> pd.DataFrame:
    """Describe whether high decision disagreement coincides with weak returns."""

    required = {
        "date",
        "daily_return",
        "drawdown",
        "turnover",
        "disagreement_to_consensus_tv",
    }
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily disagreement report is missing: {sorted(missing)}")
    frame = daily.loc[
        daily["disagreement_to_consensus_tv"].notna(), list(required)
    ].copy()
    if len(frame) < groups:
        raise ValueError("not enough rebalance decisions for disagreement groups")
    frame["disagreement_quantile"] = pd.qcut(
        frame["disagreement_to_consensus_tv"],
        q=groups,
        labels=[f"Q{index}" for index in range(1, groups + 1)],
        duplicates="drop",
    )
    report = (
        frame.groupby("disagreement_quantile", observed=True)
        .agg(
            observations=("daily_return", "size"),
            disagreement_mean=("disagreement_to_consensus_tv", "mean"),
            disagreement_min=("disagreement_to_consensus_tv", "min"),
            disagreement_max=("disagreement_to_consensus_tv", "max"),
            mean_daily_return=("daily_return", "mean"),
            compounded_return=(
                "daily_return",
                lambda values: float(np.prod(1.0 + values) - 1.0),
            ),
            mean_turnover=("turnover", "mean"),
            mean_drawdown=("drawdown", "mean"),
            worst_drawdown=("drawdown", "min"),
        )
        .reset_index()
    )
    report["return_rank"] = report["mean_daily_return"].rank(
        ascending=False, method="min"
    )
    return report


def calibrate_attached_rebalance_strategies(
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    ensemble: EnsembleConfig,
    agents: list[PPOAgent],
    *,
    seed: int,
    verbose: bool = True,
) -> tuple[
    dict[str, AdaptiveRebalancePolicy],
    pd.DataFrame,
    dict[str, object],
    pd.DataFrame,
]:
    """Select each user-supplied strategy on calibration, never on test."""

    ensemble.validate()
    mean_env = make_environment(
        data, experiment, market, split="calibration", seed=seed
    )
    mean_daily, mean_metrics = evaluate_mean_ensemble(mean_env, agents)
    decisions = mean_daily["disagreement_to_consensus_tv"].dropna().to_numpy(
        dtype=np.float64
    )
    if decisions.size < 5:
        raise ValueError("calibration needs at least five rebalance decisions")
    low_threshold = float(np.quantile(decisions, 0.25))
    high_threshold = float(np.quantile(decisions, 0.75))

    candidates: dict[str, list[AdaptiveRebalancePolicy]] = {
        SOFT_DISAGREEMENT_STRATEGY: [
            AdaptiveRebalancePolicy("soft", disagreement_k=float(k))
            for k in ensemble.soft_disagreement_k_grid
        ],
        COMBINED_ADAPTIVE_STRATEGY: [
            AdaptiveRebalancePolicy(
                "combined",
                consensus_method=consensus_method,
                disagreement_k=float(k),
                base_turnover_budget=float(budget),
                base_cost_penalty=float(global_penalty),
                asset_penalty=float(asset_penalty),
            )
            for consensus_method in ensemble.combined_consensus_methods
            for k, budget, global_penalty, asset_penalty in zip(
                ensemble.combined_k_grid,
                ensemble.combined_turnover_budget_grid,
                ensemble.combined_global_penalty_grid,
                ensemble.combined_asset_penalty_grid,
                strict=True,
            )
        ],
    }
    rows: list[dict[str, float | bool | str]] = []
    candidate_lookup: dict[tuple[str, int], AdaptiveRebalancePolicy] = {}
    for strategy, policies in candidates.items():
        for candidate_index, policy in enumerate(policies):
            env = make_environment(
                data, experiment, market, split="calibration", seed=seed
            )
            daily, metrics = evaluate_adaptive_rebalance_ensemble(
                env, agents, policy
            )
            temporal_score, worst_period_return = _temporal_validation_score(
                daily
            )
            candidate_lookup[(strategy, candidate_index)] = policy
            rows.append(
                {
                    "strategy": strategy,
                    "candidate_index": candidate_index,
                    "method": policy.method,
                    "consensus_method": policy.consensus_method,
                    "disagreement_k": policy.disagreement_k,
                    "base_turnover_budget": policy.base_turnover_budget,
                    "base_cost_penalty": policy.base_cost_penalty,
                    "asset_penalty": policy.asset_penalty,
                    "low_threshold": policy.low_threshold,
                    "high_threshold": policy.high_threshold,
                    "partial_rebalance": policy.partial_rebalance,
                    "temporal_validation_score": temporal_score,
                    "worst_validation_period_return": worst_period_return,
                    "validation_score": _ensemble_reliability_objective(
                        metrics, ppo
                    ),
                    "total_return": metrics["total_return"],
                    "cagr": metrics["cagr"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "maximum_drawdown": metrics["maximum_drawdown"],
                    "annual_turnover": metrics["annual_turnover"],
                    "total_transaction_cost": metrics[
                        "total_transaction_cost"
                    ],
                    "intervention_strength": float(
                        policy.disagreement_k
                        + 4.0 * (1.0 - policy.base_turnover_budget)
                        + 100.0 * policy.base_cost_penalty
                        + policy.asset_penalty
                        + (1.0 - policy.partial_rebalance)
                    ),
                    "selected": False,
                }
            )
            if verbose:
                print(
                    f"Adaptive validation {strategy} "
                    f"candidate={candidate_index + 1}/{len(policies)} "
                    f"score={temporal_score:.4f} "
                    f"CAGR={metrics['cagr']:.4f}",
                    flush=True,
                )
    grid = pd.DataFrame(rows)
    selected_policies: dict[str, AdaptiveRebalancePolicy] = {
        # Frozen before the expanded-grid run. Keeping it as a separately
        # named benchmark prevents silent regression and avoids pretending the
        # test-informed comparison is a fresh confirmatory selection.
        COMBINED_ADAPTIVE_V1_STRATEGY: AdaptiveRebalancePolicy(
            "combined",
            consensus_method="geometric_median",
            disagreement_k=4.0,
            base_turnover_budget=0.50,
            base_cost_penalty=0.01,
            asset_penalty=0.50,
        ),
    }
    selected_payload: dict[str, object] = {}
    for strategy in candidates:
        subset = grid[grid["strategy"] == strategy]
        best_temporal_score = float(
            subset["temporal_validation_score"].max()
        )
        near_best = subset[
            subset["temporal_validation_score"]
            >= best_temporal_score - ensemble.adaptive_selection_tolerance
        ]
        # Extremely low turnover can look attractive in one calibration year
        # but leave a stale portfolio in the next regime. Keep at least 25% of
        # mean-ensemble calibration turnover whenever a candidate permits it.
        turnover_floor = 0.25 * float(mean_metrics["annual_turnover"])
        non_stale = near_best[
            near_best["annual_turnover"] >= turnover_floor
        ]
        if not non_stale.empty:
            near_best = non_stale
        selected_index = near_best.sort_values(
            [
                "validation_score",
                "sharpe_ratio",
                "intervention_strength",
                "annual_turnover",
            ],
            ascending=[False, False, True, True],
        ).index[0]
        grid.loc[selected_index, "selected"] = True
        selected_row = grid.loc[selected_index]
        candidate = candidate_lookup[
            (strategy, int(selected_row["candidate_index"]))
        ]
        selected = replace(
            candidate,
            validation_score=float(
                selected_row["temporal_validation_score"]
            ),
        )
        selected_policies[strategy] = selected
        selected_payload[strategy] = dataclass_dict(selected)

    calibration: dict[str, object] = {
        "method": "final_three_disagreement_strategy_suite",
        "proposed_strategy": PROPOSED_STRATEGY,
        "retained_strategies": list(FINAL_OPTIMIZED_STRATEGIES),
        "uses_seed_performance_weights": False,
        "seed_rights": "equal",
        "disagreement_formula": (
            "mean_i(0.5*L1(w_i - arithmetic_mean(W)))"
        ),
        "candidate_count": len(grid),
        "selection_tolerance": ensemble.adaptive_selection_tolerance,
        "minimum_turnover_fraction_of_mean": 0.25,
        "fit_period": {
            "start": experiment.calibration_start,
            "end": experiment.calibration_end,
        },
        "test_period_not_used_for_selection": True,
        "selection_objective": (
            "retain candidates within tolerance of the best mean quarterly "
            "return - 0.5*quarterly return std; reject stale portfolios below "
            "25% of mean-ensemble turnover when possible; then choose by "
            "risk-adjusted score, Sharpe, and lower intervention complexity"
        ),
        "hard_threshold_quantiles": {
            "low_quantile": 0.25,
            "high_quantile": 0.75,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
        "selected_policies": selected_payload,
        "frozen_benchmark_policies": {
            COMBINED_ADAPTIVE_V1_STRATEGY: dataclass_dict(
                selected_policies[COMBINED_ADAPTIVE_V1_STRATEGY]
            )
        },
    }
    return (
        selected_policies,
        grid,
        calibration,
        disagreement_quantile_report(mean_daily),
    )


def evaluate_attached_rebalance_suite(
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    ensemble: EnsembleConfig,
    agents: list[PPOAgent],
    output_dir: Path,
    *,
    seed: int,
    verbose: bool = True,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, pd.DataFrame],
]:
    """Calibrate and test all strategies from the user's attached note."""

    if verbose:
        print(
            "\n=== Calibrating attached disagreement strategies ===",
            flush=True,
        )
    policies, grid, calibration, calibration_quantiles = (
        calibrate_attached_rebalance_strategies(
            data,
            experiment,
            market,
            ppo,
            ensemble,
            agents,
            seed=seed,
            verbose=verbose,
        )
    )
    grid.to_csv(
        output_dir / "adaptive_rebalance_validation_grid.csv", index=False
    )
    (output_dir / "adaptive_rebalance_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "disagreement_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    calibration_quantiles.to_csv(
        output_dir / "disagreement_quantiles_calibration.csv", index=False
    )

    results: dict[str, dict[str, float]] = {}
    histories: dict[str, pd.DataFrame] = {}
    for strategy, policy in policies.items():
        env = make_environment(
            data, experiment, market, split="test", seed=seed
        )
        daily, metrics = evaluate_adaptive_rebalance_ensemble(
            env, agents, policy
        )
        daily.to_csv(output_dir / f"daily_{strategy}.csv", index=False)
        results[strategy] = metrics
        histories[strategy] = daily
        if verbose:
            print(
                f"Selected {strategy} | method={policy.method} "
                f"consensus={policy.consensus_method} "
                f"k={policy.disagreement_k:g} "
                f"validation_score={policy.validation_score:.4f}",
                flush=True,
            )
    mean_daily_path = output_dir / "daily_PPO_mean_ensemble.csv"
    if mean_daily_path.exists():
        mean_test = pd.read_csv(mean_daily_path, parse_dates=["date"])
        disagreement_quantile_report(mean_test).to_csv(
            output_dir / "disagreement_quantiles_test.csv", index=False
        )
    return results, histories


def calibrate_disagreement_policy(
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    ensemble: EnsembleConfig,
    agents: list[PPOAgent],
    *,
    seed: int,
    verbose: bool = True,
) -> tuple[DisagreementPolicy, pd.DataFrame, dict[str, object]]:
    """Select the disagreement gate only on the ensemble-calibration split."""

    ensemble.validate()
    baseline_env = make_environment(
        data, experiment, market, split="calibration", seed=seed
    )
    baseline_daily, baseline_metrics = evaluate_mean_ensemble(
        baseline_env, agents
    )
    disagreement = baseline_daily[
        "disagreement_mean_pairwise_tv"
    ].dropna().to_numpy(dtype=np.float64)
    rows: list[dict[str, float | bool | str]] = []
    for low_quantile in ensemble.low_quantiles:
        low_threshold = float(np.quantile(disagreement, low_quantile))
        for high_quantile in ensemble.high_quantiles:
            if low_quantile >= high_quantile:
                continue
            high_threshold = float(np.quantile(disagreement, high_quantile))
            if high_threshold <= low_threshold + 1e-12:
                continue
            # Final consensus-only baseline: follow the equal-seed consensus
            # when agreement is high and keep the current portfolio once
            # disagreement reaches the upper threshold.
            for minimum_rebalance in (0.0,):
                policy = DisagreementPolicy(
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    minimum_rebalance=float(minimum_rebalance),
                    low_quantile=float(low_quantile),
                    high_quantile=float(high_quantile),
                )
                validation_env = make_environment(
                    data, experiment, market, split="calibration", seed=seed
                )
                daily, metrics = evaluate_disagreement_ensemble(
                    validation_env, agents, policy
                )
                temporal_score, worst_period_return = _temporal_validation_score(
                    daily
                )
                rows.append(
                    {
                        "strategy": CONSENSUS_ONLY_STRATEGY,
                        "low_quantile": float(low_quantile),
                        "high_quantile": float(high_quantile),
                        "low_threshold": low_threshold,
                        "high_threshold": high_threshold,
                        "minimum_rebalance": float(minimum_rebalance),
                        "validation_score": _ensemble_reliability_objective(
                            metrics, ppo
                        ),
                        "temporal_validation_score": temporal_score,
                        "worst_validation_period_return": worst_period_return,
                        "total_return": metrics["total_return"],
                        "cagr": metrics["cagr"],
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "maximum_drawdown": metrics["maximum_drawdown"],
                        "annual_turnover": metrics["annual_turnover"],
                        "total_transaction_cost": metrics[
                            "total_transaction_cost"
                        ],
                        "mean_rebalance_gate": float(
                            daily["rebalance_gate"].mean()
                        ),
                    }
                )
                if verbose:
                    print(
                        "Disagreement validation "
                        f"q=({low_quantile:.2f},{high_quantile:.2f}) "
                        f"min_gate={minimum_rebalance:.2f} "
                        f"score={rows[-1]['validation_score']:.4f}",
                        flush=True,
                    )
    if not rows:
        raise RuntimeError("No valid disagreement calibration candidates")
    grid = pd.DataFrame(rows)
    selected_index = grid.sort_values(
        [
            "temporal_validation_score",
            "validation_score",
            "sharpe_ratio",
            "annual_turnover",
        ],
        ascending=[False, False, False, True],
    ).index[0]
    grid["selected"] = False
    grid.loc[selected_index, "selected"] = True
    selected = grid.loc[selected_index]
    policy = DisagreementPolicy(
        low_threshold=float(selected["low_threshold"]),
        high_threshold=float(selected["high_threshold"]),
        minimum_rebalance=float(selected["minimum_rebalance"]),
        low_quantile=float(selected["low_quantile"]),
        high_quantile=float(selected["high_quantile"]),
        validation_score=float(selected["temporal_validation_score"]),
    )
    quantiles = {
        str(value): float(np.quantile(disagreement, value))
        for value in (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
    }
    calibration: dict[str, object] = {
        "method": "consensus_only_disagreement_gate",
        "strategy": CONSENSUS_ONLY_STRATEGY,
        "seed_aggregation": "unweighted arithmetic mean; no seed confidence weights",
        "disagreement_formula": "mean_{i<j}(0.5 * L1(w_i, w_j))",
        "target_formula": "w_target = w_current + gate(disagreement) * "
        "(mean_seed_weight - w_current)",
        "fit_period": {
            "start": experiment.calibration_start,
            "end": experiment.calibration_end,
        },
        "test_period_not_used_for_selection": True,
        "objective": {
            "formula": "mean subperiod return - 0.5*std(subperiod return); "
            "calendar quarters are used for a one-year calibration window, "
            "calendar years otherwise; "
            "tie-break by Sharpe - turnover_penalty*annual_turnover - "
            "drawdown_penalty*abs(maximum_drawdown)",
            "turnover_penalty": ppo.validation_turnover_penalty,
            "drawdown_penalty": ppo.validation_drawdown_penalty,
        },
        "validation_disagreement_quantiles": quantiles,
        "mean_ensemble_validation_metrics": baseline_metrics,
        "selected_policy": dataclass_dict(policy),
    }
    return policy, grid, calibration


def calibrate_reliability_weighted_policy(
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    ensemble: EnsembleConfig,
    agents: list[PPOAgent],
    gate_policy: DisagreementPolicy,
    *,
    seeds: list[int],
    verbose: bool = True,
) -> tuple[ReliabilityWeightedPolicy, pd.DataFrame, dict[str, object]]:
    """Select regularized seed-reliability weights on calibration only."""

    if len(agents) != len(seeds):
        raise ValueError("agents and seeds must have the same length")
    seed_metrics: list[dict[str, float]] = []
    seed_scores: list[float] = []
    for agent, agent_seed in zip(agents, seeds):
        env = make_environment(
            data, experiment, market, split="calibration", seed=agent_seed
        )
        _, metrics = evaluate_agent(env, agent)
        seed_metrics.append(metrics)
        # Seed reliability is risk-adjusted performance only. Turnover and
        # drawdown are already penalized when each ensemble candidate is
        # evaluated below; applying them here too would double-penalize costs.
        seed_scores.append(float(metrics["sharpe_ratio"]))

    score_array = np.asarray(seed_scores, dtype=np.float64)
    rows: list[dict[str, float | bool | str]] = []
    policies: dict[float, ReliabilityWeightedPolicy] = {}
    for temperature in ensemble.reliability_temperature_grid:
        logits = (score_array - score_array.max()) / float(temperature)
        raw_weights = np.exp(logits)
        weights = raw_weights / raw_weights.sum()
        policy = ReliabilityWeightedPolicy(
            seed_weights=tuple(float(value) for value in weights),
            seed_scores=tuple(float(value) for value in score_array),
            temperature=float(temperature),
            gate_policy=gate_policy,
        )
        env = make_environment(
            data, experiment, market, split="calibration", seed=seeds[0]
        )
        daily, metrics = evaluate_reliability_weighted_ensemble(
            env, agents, policy
        )
        temporal_score, worst_period_return = _temporal_validation_score(daily)
        policies[float(temperature)] = policy
        row: dict[str, float | bool | str] = {
            "strategy": RELIABILITY_WEIGHTED_STRATEGY,
            "temperature": float(temperature),
            "temporal_validation_score": temporal_score,
            "worst_validation_period_return": worst_period_return,
            "validation_score": _ensemble_reliability_objective(metrics, ppo),
            "total_return": metrics["total_return"],
            "cagr": metrics["cagr"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "maximum_drawdown": metrics["maximum_drawdown"],
            "annual_turnover": metrics["annual_turnover"],
            "total_transaction_cost": metrics["total_transaction_cost"],
            "effective_seed_count": float(1.0 / np.sum(weights**2)),
            "maximum_seed_weight": float(weights.max()),
            "selected": False,
        }
        for agent_seed, weight in zip(seeds, weights):
            row[f"weight_seed_{agent_seed}"] = float(weight)
        rows.append(row)
        if verbose:
            print(
                "Reliability ensemble validation "
                f"temperature={temperature:g} "
                f"temporal_score={temporal_score:.4f} "
                f"CAGR={metrics['cagr']:.4f}",
                flush=True,
            )

    grid = pd.DataFrame(rows)
    selected_index = grid.sort_values(
        [
            "temporal_validation_score",
            "validation_score",
            "sharpe_ratio",
            "annual_turnover",
        ],
        ascending=[False, False, False, True],
    ).index[0]
    grid.loc[selected_index, "selected"] = True
    selected_temperature = float(grid.loc[selected_index, "temperature"])
    base_policy = policies[selected_temperature]
    policy = ReliabilityWeightedPolicy(
        seed_weights=base_policy.seed_weights,
        seed_scores=base_policy.seed_scores,
        temperature=base_policy.temperature,
        gate_policy=base_policy.gate_policy,
        validation_score=float(
            grid.loc[selected_index, "temporal_validation_score"]
        ),
    )
    calibration: dict[str, object] = {
        "method": "calibration_reliability_weighted_disagreement_ensemble",
        "strategy": RELIABILITY_WEIGHTED_STRATEGY,
        "seed_order": seeds,
        "seed_score_formula": "calibration Sharpe ratio",
        "seed_scores": {
            str(agent_seed): float(score)
            for agent_seed, score in zip(seeds, score_array)
        },
        "seed_calibration_metrics": {
            str(agent_seed): metrics
            for agent_seed, metrics in zip(seeds, seed_metrics)
        },
        "weight_formula": "softmax((score_i - max(score))/temperature)",
        "global_disagreement_gate": dataclass_dict(gate_policy),
        "fit_period": {
            "start": experiment.calibration_start,
            "end": experiment.calibration_end,
        },
        "test_period_not_used_for_selection": True,
        "selected_policy": {
            "temperature": policy.temperature,
            "seed_weights": {
                str(agent_seed): float(weight)
                for agent_seed, weight in zip(seeds, policy.seed_weights)
            },
            "validation_score": policy.validation_score,
        },
    }
    return policy, grid, calibration


def calibrate_directional_disagreement_policy(
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    ensemble: EnsembleConfig,
    agents: list[PPOAgent],
    gate_policy: DisagreementPolicy,
    *,
    seed: int,
    verbose: bool = True,
) -> tuple[DirectionalDisagreementPolicy, pd.DataFrame, dict[str, object]]:
    """Tune directional trade shrinkage on the disjoint calibration split.

    The deliberately small second-stage grid limits validation overfitting.  It
    includes ``kappa=eta=0``, which is exactly the fixed adaptive mean policy,
    so the optimizer must earn any additional complexity on calibration data.
    """

    rows: list[dict[str, float | bool | str]] = []
    for shrinkage in ensemble.directional_shrinkage_grid:
        for turnover_regularization in (
            ensemble.directional_turnover_regularization_grid
        ):
            policy = DirectionalDisagreementPolicy(
                low_threshold=gate_policy.low_threshold,
                high_threshold=gate_policy.high_threshold,
                minimum_rebalance=gate_policy.minimum_rebalance,
                low_quantile=gate_policy.low_quantile,
                high_quantile=gate_policy.high_quantile,
                shrinkage_coefficient=float(shrinkage),
                turnover_regularization=float(turnover_regularization),
            )
            env = make_environment(
                data, experiment, market, split="calibration", seed=seed
            )
            daily, metrics = evaluate_directional_disagreement_ensemble(
                env, agents, policy
            )
            temporal_score, worst_year_return = _temporal_validation_score(
                daily
            )
            review_days = daily["disagreement_mean_pairwise_tv"].notna()
            rows.append(
                {
                    "strategy": EQUAL_SEED_DISAGREEMENT_STRATEGY,
                    "shrinkage_coefficient": float(shrinkage),
                    "turnover_regularization": float(turnover_regularization),
                    "low_threshold": gate_policy.low_threshold,
                    "high_threshold": gate_policy.high_threshold,
                    "minimum_rebalance": gate_policy.minimum_rebalance,
                    "temporal_validation_score": temporal_score,
                    "worst_validation_year_return": worst_year_return,
                    "validation_score": _ensemble_reliability_objective(
                        metrics, ppo
                    ),
                    "total_return": metrics["total_return"],
                    "cagr": metrics["cagr"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "maximum_drawdown": metrics["maximum_drawdown"],
                    "annual_turnover": metrics["annual_turnover"],
                    "total_transaction_cost": metrics[
                        "total_transaction_cost"
                    ],
                    "mean_shrunk_trade_size": float(
                        daily.loc[
                            review_days, "directional_shrunk_trade_size"
                        ].mean()
                    ),
                    "mean_suppressed_assets": float(
                        daily.loc[
                            review_days, "directional_suppressed_assets"
                        ].mean()
                    ),
                    "selected": False,
                }
            )
            if verbose:
                print(
                    "Directional disagreement validation "
                    f"kappa={shrinkage:g} eta={turnover_regularization:g} "
                    f"temporal_score={temporal_score:.4f} "
                    f"CAGR={metrics['cagr']:.4f}",
                    flush=True,
                )
    if not rows:
        raise RuntimeError("No directional disagreement candidates")
    grid = pd.DataFrame(rows)
    ranked = grid.sort_values(
        [
            "temporal_validation_score",
            "validation_score",
            "sharpe_ratio",
            "annual_turnover",
        ],
        ascending=[False, False, False, True],
    )
    raw_best_index = ranked.index[0]
    fallback = grid[
        np.isclose(grid["shrinkage_coefficient"], 0.0)
        & np.isclose(grid["turnover_regularization"], 0.0)
    ]
    if fallback.empty:
        raise ValueError(
            "directional grids must include kappa=0 and eta=0 as fallback"
        )
    fallback_index = fallback.index[0]
    raw_improvement = float(
        grid.loc[raw_best_index, "temporal_validation_score"]
        - grid.loc[fallback_index, "temporal_validation_score"]
    )
    complexity_guard_triggered = (
        raw_best_index != fallback_index
        and raw_improvement
        < ensemble.directional_minimum_validation_improvement
    )
    selected_index = (
        fallback_index if complexity_guard_triggered else raw_best_index
    )
    grid["raw_validation_improvement_over_fallback"] = raw_improvement
    grid["complexity_guard_triggered"] = complexity_guard_triggered
    grid["raw_best"] = False
    grid.loc[raw_best_index, "raw_best"] = True
    grid.loc[selected_index, "selected"] = True
    selected = grid.loc[selected_index]
    policy = DirectionalDisagreementPolicy(
        low_threshold=gate_policy.low_threshold,
        high_threshold=gate_policy.high_threshold,
        minimum_rebalance=gate_policy.minimum_rebalance,
        low_quantile=gate_policy.low_quantile,
        high_quantile=gate_policy.high_quantile,
        shrinkage_coefficient=float(selected["shrinkage_coefficient"]),
        turnover_regularization=float(selected["turnover_regularization"]),
        validation_score=float(selected["temporal_validation_score"]),
    )
    calibration: dict[str, object] = {
        "method": "directional_equal_seed_disagreement_optimization",
        "paper_name": "Directional Disagreement-Regularized Rebalancing (D3R)",
        "uses_seed_performance_weights": False,
        "seed_rights": "equal",
        "selection_is_two_stage": True,
        "fixed_global_disagreement_gate": dataclass_dict(gate_policy),
        "candidate_count": len(grid),
        "complexity_guard": {
            "fallback": "kappa=0 and eta=0 (fixed TC-EDR gate)",
            "minimum_temporal_validation_improvement": (
                ensemble.directional_minimum_validation_improvement
            ),
            "raw_best_improvement_over_fallback": raw_improvement,
            "triggered": bool(complexity_guard_triggered),
        },
        "objective": {
            "seed_trade": "delta_i = w_i - w_current",
            "closed_form_before_projection": (
                "delta_star = soft_threshold(mean_i(delta_i), "
                "kappa*std_i(delta_i)) / (1+eta)"
            ),
            "portfolio_constraint": "simplex projection: w>=0 and sum(w)=1",
            "global_control": (
                "validation-calibrated mean pairwise TV gate controls the "
                "remaining movement toward the optimized portfolio"
            ),
        },
        "fit_period": {
            "start": experiment.calibration_start,
            "end": experiment.calibration_end,
        },
        "test_period_not_used_for_selection": True,
        "selection_objective": (
            "mean annual validation return minus 0.5 times annual standard "
            "deviation; then risk-adjusted score and lower turnover"
        ),
        "selected_policy": dataclass_dict(policy),
    }
    return policy, grid, calibration


def calibrate_regime_disagreement_policy(
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    ensemble: EnsembleConfig,
    agents: list[PPOAgent],
    gate_policy: DisagreementPolicy,
    *,
    seed: int,
    verbose: bool = True,
) -> tuple[RegimeDisagreementPolicy, pd.DataFrame, dict[str, object]]:
    """Select only the VNINDEX risk-off exposure after fixing the seed gate."""

    rows: list[dict[str, float | bool | str]] = []
    for risk_off_exposure in ensemble.regime_risk_off_exposure_grid:
        policy = RegimeDisagreementPolicy(
            low_threshold=gate_policy.low_threshold,
            high_threshold=gate_policy.high_threshold,
            minimum_rebalance=gate_policy.minimum_rebalance,
            low_quantile=gate_policy.low_quantile,
            high_quantile=gate_policy.high_quantile,
            trend_lookback_days=ensemble.regime_lookback_days,
            risk_off_stock_exposure=float(risk_off_exposure),
        )
        env = make_environment(
            data, experiment, market, split="calibration", seed=seed
        )
        daily, metrics = evaluate_regime_disagreement_ensemble(
            env, agents, policy
        )
        temporal_score, worst_year_return = _temporal_validation_score(daily)
        review_days = daily["market_risk_off"].notna()
        rows.append(
            {
                "strategy": "PPO_vnindex_regime_overlay_ablation",
                "trend_lookback_days": ensemble.regime_lookback_days,
                "risk_off_stock_exposure": float(risk_off_exposure),
                "temporal_validation_score": temporal_score,
                "worst_validation_year_return": worst_year_return,
                "validation_score": _ensemble_reliability_objective(
                    metrics, ppo
                ),
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "annual_turnover": metrics["annual_turnover"],
                "total_transaction_cost": metrics["total_transaction_cost"],
                "risk_off_review_fraction": float(
                    daily.loc[review_days, "market_risk_off"].mean()
                ),
                "selected": False,
            }
        )
        if verbose:
            print(
                "Regime disagreement validation "
                f"risk_off_exposure={risk_off_exposure:.2f} "
                f"temporal_score={temporal_score:.4f}",
                flush=True,
            )
    grid = pd.DataFrame(rows)
    selected_index = grid.sort_values(
        [
            "temporal_validation_score",
            "validation_score",
            "sharpe_ratio",
            "annual_turnover",
        ],
        ascending=[False, False, False, True],
    ).index[0]
    grid.loc[selected_index, "selected"] = True
    selected = grid.loc[selected_index]
    policy = RegimeDisagreementPolicy(
        low_threshold=gate_policy.low_threshold,
        high_threshold=gate_policy.high_threshold,
        minimum_rebalance=gate_policy.minimum_rebalance,
        low_quantile=gate_policy.low_quantile,
        high_quantile=gate_policy.high_quantile,
        trend_lookback_days=int(selected["trend_lookback_days"]),
        risk_off_stock_exposure=float(selected["risk_off_stock_exposure"]),
        validation_score=float(selected["temporal_validation_score"]),
    )
    calibration: dict[str, object] = {
        "method": "regime_constrained_equal_seed_disagreement_rebalancing",
        "uses_seed_performance_weights": False,
        "seed_rights": "equal",
        "selection_is_two_stage": True,
        "fixed_disagreement_gate": dataclass_dict(gate_policy),
        "market_signal": (
            "VNINDEX close-to-close return over the preceding fully observed "
            f"{ensemble.regime_lookback_days} trading days; risk-off below zero"
        ),
        "fit_period": {
            "start": experiment.calibration_start,
            "end": experiment.calibration_end,
        },
        "test_period_not_used_for_selection": True,
        "selection_objective": (
            "mean annual validation return minus 0.5 times annual standard "
            "deviation; then risk-adjusted score and lower turnover"
        ),
        "selected_policy": dataclass_dict(policy),
    }
    return policy, grid, calibration


def calibrate_optimized_disagreement_policy(
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    ensemble: EnsembleConfig,
    agents: list[PPOAgent],
    gate_policy: DisagreementPolicy,
    *,
    seed: int,
    verbose: bool = True,
) -> tuple[DisagreementOptimizationPolicy, pd.DataFrame, dict[str, object]]:
    """Tune a convex cross-seed optimizer on the calibration split only."""

    ensemble.validate()
    rows: list[dict[str, float | bool | str]] = []

    # The global gate is fixed in the first calibration stage. This stage tunes
    # only the asset-level disagreement term and turnover regularization, which
    # keeps the search small and avoids fitting the same gate twice.
    for consensus_method in (ensemble.optimization_primary_consensus_method,):
        for disagreement_coefficient in (
            ensemble.optimization_disagreement_coefficient_grid
        ):
            for turnover_regularization in (
                ensemble.optimization_turnover_regularization_grid
            ):
                policy = DisagreementOptimizationPolicy(
                    disagreement_coefficient=float(disagreement_coefficient),
                    turnover_regularization=float(turnover_regularization),
                    consensus_method=consensus_method,
                    low_threshold=gate_policy.low_threshold,
                    high_threshold=gate_policy.high_threshold,
                    minimum_rebalance=gate_policy.minimum_rebalance,
                )
                env = make_environment(
                    data, experiment, market, split="calibration", seed=seed
                )
                daily, metrics = evaluate_optimized_disagreement_ensemble(
                    env, agents, policy
                )
                temporal_score, worst_year_return = _temporal_validation_score(
                    daily
                )
                review_days = daily[
                    "disagreement_mean_pairwise_tv"
                ].notna()
                rows.append(
                    {
                        "strategy": CROSS_SEED_OPTIMIZER_STRATEGY,
                        "consensus_method": consensus_method,
                        "disagreement_coefficient": float(
                            disagreement_coefficient
                        ),
                        "turnover_regularization": float(
                            turnover_regularization
                        ),
                        "low_quantile": gate_policy.low_quantile,
                        "high_quantile": gate_policy.high_quantile,
                        "low_threshold": gate_policy.low_threshold,
                        "high_threshold": gate_policy.high_threshold,
                        "minimum_rebalance": gate_policy.minimum_rebalance,
                        "total_return": metrics["total_return"],
                        "cagr": metrics["cagr"],
                        "temporal_validation_score": temporal_score,
                        "worst_validation_year_return": worst_year_return,
                        "sharpe_ratio": metrics["sharpe_ratio"],
                        "maximum_drawdown": metrics["maximum_drawdown"],
                        "annual_turnover": metrics["annual_turnover"],
                        "total_transaction_cost": metrics[
                            "total_transaction_cost"
                        ],
                        "mean_disagreement": float(
                            daily.loc[
                                review_days,
                                "disagreement_mean_pairwise_tv",
                            ].mean()
                        ),
                        "mean_optimized_uncertainty_exposure": float(
                            daily.loc[
                                review_days,
                                "optimized_uncertainty_exposure",
                            ].mean()
                        ),
                        "selected": False,
                    }
                )
                if verbose:
                    print(
                        "Cross-seed optimizer validation "
                        f"kappa={disagreement_coefficient:g} "
                        f"eta={turnover_regularization:g} "
                        f"temporal_score={temporal_score:.4f} "
                        f"CAGR={metrics['cagr']:.4f}",
                        flush=True,
                    )
    if not rows:
        raise RuntimeError("No disagreement optimization candidates")
    grid = pd.DataFrame(rows)
    primary_candidates = grid[
        grid["consensus_method"]
        == ensemble.optimization_primary_consensus_method
    ]
    if primary_candidates.empty:
        raise RuntimeError("primary consensus method has no validation candidates")
    # kappa=0 is retained in the grid as an auditable no-disagreement baseline,
    # but it is not eligible to represent this explicitly cross-seed strategy.
    distinct_candidates = primary_candidates[
        ~np.isclose(primary_candidates["disagreement_coefficient"], 0.0)
    ]
    if distinct_candidates.empty:
        raise ValueError(
            "optimization grid must contain a non-zero disagreement coefficient"
        )
    selected_index = distinct_candidates.sort_values(
        [
            "temporal_validation_score",
            "cagr",
            "sharpe_ratio",
            "annual_turnover",
        ],
        ascending=[False, False, False, True],
    ).index[0]
    grid.loc[selected_index, "selected"] = True
    selected = grid.loc[selected_index]
    policy = DisagreementOptimizationPolicy(
        disagreement_coefficient=float(
            selected["disagreement_coefficient"]
        ),
        turnover_regularization=float(selected["turnover_regularization"]),
        consensus_method=str(selected["consensus_method"]),
        low_threshold=float(selected["low_threshold"]),
        high_threshold=float(selected["high_threshold"]),
        minimum_rebalance=float(selected["minimum_rebalance"]),
        validation_cagr=float(selected["cagr"]),
    )
    calibration: dict[str, object] = {
        "method": "convex_cross_seed_disagreement_optimization",
        "strategy": CROSS_SEED_OPTIMIZER_STRATEGY,
        "uses_seed_performance_weights": False,
        "seed_rights": "equal",
        "candidate_count": len(grid),
        "zero_disagreement_candidates_are_baseline_only": True,
        "objective": {
            "consensus": (
                "fixed permutation-invariant arithmetic consensus"
            ),
            "robust_target": (
                "min 0.5*||w-consensus(W)||^2 + kappa*std(W)'w + "
                "0.5*eta*||w-w_current||^2, subject to w in simplex"
            ),
            "global_disagreement_control": (
                "w_final = w_current + gate(mean_pairwise_TV) * "
                "(w_robust - w_current)"
            ),
        },
        "fit_period": {
            "start": experiment.calibration_start,
            "end": experiment.calibration_end,
        },
        "test_period_not_used_for_selection": True,
        "ablation_consensus_methods": list(
            ensemble.optimization_consensus_methods
        ),
        "primary_consensus_method": (
            ensemble.optimization_primary_consensus_method
        ),
        "fixed_global_disagreement_gate": dataclass_dict(gate_policy),
        "selection_objective": (
            "among non-zero disagreement coefficients within the fixed "
            "primary method: mean annual validation return minus 0.5 times "
            "its annual standard deviation; then CAGR, Sharpe, and lower "
            "turnover; kappa=0 is reported only as a no-disagreement baseline"
        ),
        "selected_policy": dataclass_dict(policy),
    }
    return policy, grid, calibration


def consensus_ablation_policies(
    primary_policy: DisagreementOptimizationPolicy,
    consensus_methods: tuple[str, ...],
) -> dict[str, DisagreementOptimizationPolicy]:
    """Hold every tuned control fixed and swap only the consensus operator."""

    primary_policy.validate()
    policies: dict[str, DisagreementOptimizationPolicy] = {}
    for method in consensus_methods:
        policies[method] = DisagreementOptimizationPolicy(
            disagreement_coefficient=primary_policy.disagreement_coefficient,
            turnover_regularization=primary_policy.turnover_regularization,
            consensus_method=method,
            low_threshold=primary_policy.low_threshold,
            high_threshold=primary_policy.high_threshold,
            minimum_rebalance=primary_policy.minimum_rebalance,
            validation_cagr=primary_policy.validation_cagr,
        )
    return policies


def _evaluate_baselines_and_finalize(
    *,
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    output_dir: Path,
    results: dict[str, dict[str, float]],
    histories: dict[str, pd.DataFrame],
    experiment_started: float,
    seed_timings: dict[str, dict[str, float]],
    verbose: bool,
) -> pd.DataFrame:
    """Evaluate fixed baselines and persist one proposal-aligned report."""

    for name, interval in {
        "EqualWeight_daily": 1,
        "EqualWeight_weekly": 5,
        "EqualWeight_monthly": 20,
    }.items():
        env = make_environment(data, experiment, market, split="test", seed=42)
        daily, metrics = evaluate_equal_weight(env, rebalance_every=interval)
        daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
        results[name] = metrics
        histories[name] = daily

    buy_hold_env = make_environment(data, experiment, market, split="test", seed=42)
    daily, metrics = evaluate_equal_weight_buy_and_hold(buy_hold_env)
    daily.to_csv(output_dir / "daily_EqualWeight_buy_hold.csv", index=False)
    results["EqualWeight_buy_hold"] = metrics
    histories["EqualWeight_buy_hold"] = daily

    momentum_env = make_environment(data, experiment, market, split="test", seed=42)
    daily, metrics = evaluate_cross_sectional_momentum(
        momentum_env,
        lookback_days=60,
        top_k=15,
        rebalance_every=20,
        cash_weight=0.05,
    )
    momentum_name = "Momentum60_Top15_Monthly"
    daily.to_csv(output_dir / f"daily_{momentum_name}.csv", index=False)
    results[momentum_name] = metrics
    histories[momentum_name] = daily

    inverse_vol_env = make_environment(data, experiment, market, split="test", seed=42)
    daily, metrics = evaluate_inverse_volatility(
        inverse_vol_env,
        lookback_days=60,
        rebalance_every=20,
        cash_weight=0.05,
    )
    inverse_vol_name = "InverseVolatility60_Monthly"
    daily.to_csv(output_dir / f"daily_{inverse_vol_name}.csv", index=False)
    results[inverse_vol_name] = metrics
    histories[inverse_vol_name] = daily
    results["VNINDEX"] = _vnindex_row(metrics)

    _save_paired_bootstrap_comparisons(histories, PROPOSED_STRATEGY, output_dir)
    table = metrics_table(results)
    table.to_csv(output_dir / "metrics.csv")
    save_equity_plot(histories, output_dir / "equity_curves.png")

    elapsed = time.perf_counter() - experiment_started
    timing_summary = {
        "total_experiment_seconds": elapsed,
        "total_experiment_display": format_duration(elapsed),
        "seeds": seed_timings,
    }
    (output_dir / "timing_summary.json").write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2), encoding="utf-8"
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
        print("\n=== Out-of-sample metrics ===")
        print(table.reindex(columns=columns).round(4).to_string())
        print(f"\nTotal experiment time: {format_duration(elapsed)} ({elapsed:.1f}s)")
        print(f"\nSaved experiment to: {output_dir.resolve()}")
    return table


def _evaluate_final_strategy_suite(
    *,
    data: MarketDataBundle,
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    ensemble: EnsembleConfig,
    agents: list[PPOAgent],
    seeds: list[int],
    output_dir: Path,
    results: dict[str, dict[str, float]],
    histories: dict[str, pd.DataFrame],
    started: float,
    timing_key: str,
    verbose: bool,
) -> pd.DataFrame:
    """Evaluate only the compact final research suite."""

    mean_env = make_environment(
        data, experiment, market, split="test", seed=seeds[0]
    )
    mean_daily, mean_metrics = evaluate_mean_ensemble(mean_env, agents)
    mean_name = "PPO_mean_ensemble"
    mean_daily.to_csv(output_dir / f"daily_{mean_name}.csv", index=False)
    results[mean_name] = mean_metrics
    histories[mean_name] = mean_daily

    optimized_results, optimized_histories = evaluate_attached_rebalance_suite(
        data,
        experiment,
        market,
        ppo,
        ensemble,
        agents,
        output_dir,
        seed=seeds[0],
        verbose=verbose,
    )
    unexpected = set(optimized_results) - set(FINAL_OPTIMIZED_STRATEGIES)
    if unexpected:
        raise RuntimeError(f"Unexpected final strategies: {sorted(unexpected)}")
    results.update(optimized_results)
    histories.update(optimized_histories)

    if verbose:
        print("\n=== Calibrating consensus-only gate ===", flush=True)
    consensus_policy, consensus_grid, consensus_calibration = (
        calibrate_disagreement_policy(
            data,
            experiment,
            market,
            ppo,
            ensemble,
            agents,
            seed=seeds[0],
            verbose=verbose,
        )
    )
    consensus_grid.to_csv(
        output_dir / "consensus_only_validation_grid.csv", index=False
    )
    (output_dir / "consensus_only_calibration.json").write_text(
        json.dumps(consensus_calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    consensus_env = make_environment(
        data, experiment, market, split="test", seed=seeds[0]
    )
    consensus_daily, consensus_metrics = evaluate_disagreement_ensemble(
        consensus_env, agents, consensus_policy
    )
    consensus_daily.to_csv(
        output_dir / f"daily_{CONSENSUS_ONLY_STRATEGY}.csv", index=False
    )
    results[CONSENSUS_ONLY_STRATEGY] = consensus_metrics
    histories[CONSENSUS_ONLY_STRATEGY] = consensus_daily

    equal_name = "EqualWeight_monthly"
    equal_env = make_environment(
        data, experiment, market, split="test", seed=42
    )
    equal_daily, equal_metrics = evaluate_equal_weight(
        equal_env, rebalance_every=20
    )
    equal_daily.to_csv(output_dir / f"daily_{equal_name}.csv", index=False)
    results[equal_name] = equal_metrics
    histories[equal_name] = equal_daily
    results["VNINDEX"] = _vnindex_row(equal_metrics)

    retained = [
        *(f"PPO_seed_{seed}" for seed in seeds),
        *FINAL_OPTIMIZED_STRATEGIES,
        mean_name,
        CONSENSUS_ONLY_STRATEGY,
        equal_name,
        "VNINDEX",
    ]
    missing = set(retained) - set(results)
    if missing:
        raise RuntimeError(f"Final report is missing: {sorted(missing)}")
    results = {name: results[name] for name in retained}
    histories = {
        name: histories[name]
        for name in retained
        if name in histories
    }

    _save_paired_bootstrap_comparisons(
        histories, PROPOSED_STRATEGY, output_dir
    )
    table = metrics_table(results)
    table.to_csv(output_dir / "metrics.csv")
    save_equity_plot(histories, output_dir / "equity_curves.png")
    (output_dir / "final_strategy_manifest.json").write_text(
        json.dumps(
            {
                "ppo_seeds": seeds,
                "optimized_strategies": list(FINAL_OPTIMIZED_STRATEGIES),
                "baselines": [
                    mean_name,
                    CONSENSUS_ONLY_STRATEGY,
                    equal_name,
                    "VNINDEX",
                ],
                "consensus_only_rule": (
                    "trade toward the equal-seed mean below the low "
                    "disagreement threshold; keep the current portfolio "
                    "at or above the high threshold"
                ),
                "test_period_not_used_for_calibration": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    timing_path = output_dir / "timing_summary.json"
    timing = (
        json.loads(timing_path.read_text(encoding="utf-8"))
        if timing_path.exists()
        else {}
    )
    timing[timing_key] = elapsed
    timing[f"{timing_key}_display"] = format_duration(elapsed)
    timing["final_strategy_count"] = len(table)
    timing_path.write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
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
        print("\n=== Final compact metrics ===")
        print(table.reindex(columns=columns).round(4).to_string())
        print(f"\nFinal evaluation time: {format_duration(elapsed)}")
        print(f"Saved experiment to: {output_dir.resolve()}")
    return table

def run_experiment(
    experiment: ExperimentConfig,
    market: MarketConfig,
    ppo: PPOConfig,
    *,
    seeds: list[int],
    ensemble: EnsembleConfig | None = None,
    evaluation_interval: int = 2,
    train_only: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    experiment_started = time.perf_counter()
    experiment.validate()
    market.validate()
    ppo.validate()
    ensemble = ensemble or EnsembleConfig()
    ensemble.validate()
    seeds = list(validate_research_seeds(seeds))
    device_info = training_device_info(ppo.device)
    if verbose:
        print(f"\n{format_device_banner(device_info)}", flush=True)

    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = {
        "prices_sha256": file_sha256(experiment.data_path),
        "universe_sha256": file_sha256(experiment.universe_path),
    }
    data = load_market_data(
        experiment.data_path,
        train_start=experiment.train_start,
        train_end=experiment.train_end,
        lookback=experiment.lookback,
        minimum_training_days=experiment.minimum_training_days,
        membership_path=experiment.universe_path,
        require_point_in_time_universe=experiment.require_point_in_time_universe,
        required_end=experiment.test_end,
        expected_universe_size=experiment.expected_universe_size,
    )
    if input_files != {
        "prices_sha256": file_sha256(experiment.data_path),
        "universe_sha256": file_sha256(experiment.universe_path),
    }:
        raise RuntimeError("A research input file changed while it was being loaded")
    configuration = {
        "backtest_protocol_version": BACKTEST_PROTOCOL_VERSION,
        "model_version": MODEL_VERSION,
        "training_algorithm_version": TRAINING_ALGORITHM_VERSION,
        "training_device": device_info,
        "experiment": dataclass_dict(experiment),
        "market": dataclass_dict(market),
        "ppo": dataclass_dict(ppo),
        "ensemble": dataclass_dict(ensemble),
        "evaluation_interval": evaluation_interval,
        "seeds": seeds,
        "proposed_strategy": PROPOSED_STRATEGY,
        "final_optimized_strategies": list(FINAL_OPTIMIZED_STRATEGIES),
        "consensus_only_strategy": CONSENSUS_ONLY_STRATEGY,
        "input_files": input_files,
        "tickers": list(data.tickers),
        "eligible_tickers": [
            ticker
            for ticker, eligible in zip(data.tickers, data.eligible_assets)
            if eligible
        ],
        "excluded_tickers": [
            ticker
            for ticker, eligible in zip(data.tickers, data.eligible_assets)
            if not eligible
        ],
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results: dict[str, dict[str, float]] = {}
    histories: dict[str, pd.DataFrame] = {}
    agents: list[PPOAgent] = []
    seed_timings: dict[str, dict[str, float]] = {}
    completed_seed_seconds: list[float] = []

    for seed_index, seed in enumerate(seeds, start=1):
        seed_started = time.perf_counter()
        if verbose:
            print(
                f"\n=== Training PPO seed {seed} "
                f"({seed_index}/{len(seeds)}) ===",
                flush=True,
            )
        seed_config = replace(ppo, seed=seed)
        train_env = make_environment(
            data, experiment, market, split="train", seed=seed
        )
        validation_env = make_environment(
            data, experiment, market, split="validation", seed=seed
        )
        agent = PPOAgent(train_env.spec, seed_config)
        training_history = agent.train(
            train_env,
            validation_env=validation_env,
            evaluation_interval=evaluation_interval,
            verbose=verbose,
        )
        training_seconds = float(training_history[-1]["elapsed_seconds"])
        pd.DataFrame(training_history).to_csv(
            output_dir / f"training_seed_{seed}.csv", index=False
        )
        agent.save(output_dir / f"ppo_seed_{seed}.pt")
        if not train_only:
            test_env = make_environment(
                data, experiment, market, split="test", seed=seed
            )
            daily, metrics = evaluate_agent(test_env, agent)
            name = f"PPO_seed_{seed}"
            daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
            results[name] = metrics
            histories[name] = daily
        agents.append(agent)
        seed_total_seconds = time.perf_counter() - seed_started
        completed_seed_seconds.append(seed_total_seconds)
        remaining_seed_seconds = float(
            np.mean(completed_seed_seconds) * (len(seeds) - seed_index)
        )
        seed_timings[str(seed)] = {
            "training_seconds": training_seconds,
            "total_seed_seconds": seed_total_seconds,
        }
        if verbose:
            print(
                f"Seed {seed} complete | train={format_duration(training_seconds)} "
                f"| train+test={format_duration(seed_total_seconds)} "
                f"| estimated remaining seeds="
                f"{format_duration(remaining_seed_seconds)}",
                flush=True,
            )

    configuration["checkpoint_files"] = {
        str(seed): {
            "filename": f"ppo_seed_{seed}.pt",
            "sha256": file_sha256(output_dir / f"ppo_seed_{seed}.pt"),
        }
        for seed in seeds
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if train_only:
        elapsed = time.perf_counter() - experiment_started
        timing_summary = {
            "training_stage_seconds": elapsed,
            "training_stage_display": format_duration(elapsed),
            "seeds": seed_timings,
            "ensemble_optimization_completed": False,
        }
        (output_dir / "timing_summary.json").write_text(
            json.dumps(timing_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if verbose:
            print(
                "\nSaved 5 PPO checkpoints. Run optimize_ensemble.py "
                "to calibrate and evaluate the ensemble.",
                flush=True,
            )
        return pd.DataFrame.from_dict(seed_timings, orient="index")

    return _evaluate_final_strategy_suite(
        data=data,
        experiment=experiment,
        market=market,
        ppo=ppo,
        ensemble=ensemble,
        agents=agents,
        seeds=seeds,
        output_dir=output_dir,
        results=results,
        histories=histories,
        started=experiment_started,
        timing_key="total_experiment_seconds",
        verbose=verbose,
    )

    if len(agents) > 1:
        if verbose:
            print(
                "\n=== Calibrating disagreement strategy on disjoint calibration ===",
                flush=True,
            )
        policy, calibration_grid, calibration = calibrate_disagreement_policy(
            data,
            experiment,
            market,
            ppo,
            ensemble,
            agents,
            seed=seeds[0],
            verbose=verbose,
        )
        calibration_grid.to_csv(
            output_dir / "disagreement_validation_grid.csv", index=False
        )
        (output_dir / "disagreement_calibration.json").write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Preserve the scalar gate as an explicit ablation. The proposed
        # strategy is selected by reliability calibration below.
        calibration_grid.to_csv(
            output_dir / "scalar_gate_validation_grid.csv", index=False
        )
        (output_dir / "scalar_gate_calibration.json").write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if verbose:
            print(
                "Selected disagreement policy | "
                f"q=({policy.low_quantile:.2f},{policy.high_quantile:.2f}) "
                f"thresholds=({policy.low_threshold:.4f},"
                f"{policy.high_threshold:.4f}) "
                f"min_gate={policy.minimum_rebalance:.2f} "
                f"validation_score={policy.validation_score:.4f}",
                flush=True,
            )

        if verbose:
            print(
                "\n=== Calibrating primary reliability-weighted strategy ===",
                flush=True,
            )
        (
            reliability_policy,
            reliability_grid,
            reliability_calibration,
        ) = calibrate_reliability_weighted_policy(
            data,
            experiment,
            market,
            ppo,
            ensemble,
            agents,
            policy,
            seeds=seeds,
            verbose=verbose,
        )
        reliability_grid.to_csv(
            output_dir / "reliability_weighted_validation_grid.csv",
            index=False,
        )
        reliability_grid.to_csv(
            output_dir / "disagreement_validation_grid.csv", index=False
        )
        reliability_payload = json.dumps(
            reliability_calibration, ensure_ascii=False, indent=2
        )
        (output_dir / "reliability_weighted_calibration.json").write_text(
            reliability_payload, encoding="utf-8"
        )
        (output_dir / "disagreement_calibration.json").write_text(
            reliability_payload, encoding="utf-8"
        )

        mean_ensemble_env = make_environment(
            data, experiment, market, split="test", seed=seeds[0]
        )
        daily, metrics = evaluate_mean_ensemble(mean_ensemble_env, agents)
        name = "PPO_mean_ensemble"
        daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
        results[name] = metrics
        histories[name] = daily

        attached_results, attached_histories = (
            evaluate_attached_rebalance_suite(
                data,
                experiment,
                market,
                ppo,
                ensemble,
                agents,
                output_dir,
                seed=seeds[0],
                verbose=verbose,
            )
        )
        results.update(attached_results)
        histories.update(attached_histories)

        scalar_gate_env = make_environment(
            data, experiment, market, split="test", seed=seeds[0]
        )
        daily, metrics = evaluate_disagreement_ensemble(
            scalar_gate_env, agents, policy
        )
        name = "PPO_scalar_disagreement_gate_ablation"
        daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
        results[name] = metrics
        histories[name] = daily

        proposed_env = make_environment(
            data, experiment, market, split="test", seed=seeds[0]
        )
        daily, metrics = evaluate_reliability_weighted_ensemble(
            proposed_env, agents, reliability_policy
        )
        name = RELIABILITY_WEIGHTED_STRATEGY
        daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
        results[name] = metrics
        histories[name] = daily
        if verbose:
            print(
                "Selected reliability policy | "
                f"temperature={reliability_policy.temperature:g} "
                f"weights={np.round(reliability_policy.seed_weights, 3).tolist()} "
                f"validation_score={reliability_policy.validation_score:.4f}",
                flush=True,
            )

        if verbose:
            print(
                "\n=== Calibrating directional disagreement optimizer ===",
                flush=True,
            )
        (
            directional_policy,
            directional_grid,
            directional_calibration,
        ) = calibrate_directional_disagreement_policy(
            data,
            experiment,
            market,
            ppo,
            ensemble,
            agents,
            policy,
            seed=seeds[0],
            verbose=verbose,
        )
        directional_grid.to_csv(
            output_dir / "directional_disagreement_validation_grid.csv",
            index=False,
        )
        (output_dir / "directional_disagreement_calibration.json").write_text(
            json.dumps(
                directional_calibration, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        directional_env = make_environment(
            data, experiment, market, split="test", seed=seeds[0]
        )
        daily, metrics = evaluate_directional_disagreement_ensemble(
            directional_env, agents, directional_policy
        )
        name = EQUAL_SEED_DISAGREEMENT_STRATEGY
        daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
        results[name] = metrics
        histories[name] = daily
        if verbose:
            print(
                "Selected D3R policy | "
                f"kappa={directional_policy.shrinkage_coefficient:g} "
                f"eta={directional_policy.turnover_regularization:g} "
                f"validation_score={directional_policy.validation_score:.4f}",
                flush=True,
            )

        if verbose:
            print(
                "\n=== Calibrating convex cross-seed optimizer ===",
                flush=True,
            )
        (
            optimized_policy,
            optimization_grid,
            optimization_calibration,
        ) = calibrate_optimized_disagreement_policy(
            data,
            experiment,
            market,
            ensemble,
            agents,
            policy,
            seed=seeds[0],
            verbose=verbose,
        )
        optimization_grid.to_csv(
            output_dir / "disagreement_optimization_validation_grid.csv",
            index=False,
        )
        (output_dir / "disagreement_optimization_calibration.json").write_text(
            json.dumps(
                optimization_calibration, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        optimized_env = make_environment(
            data, experiment, market, split="test", seed=seeds[0]
        )
        daily, metrics = evaluate_optimized_disagreement_ensemble(
            optimized_env, agents, optimized_policy
        )
        name = CROSS_SEED_OPTIMIZER_STRATEGY
        daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
        results[name] = metrics
        histories[name] = daily
        if verbose:
            print(
                "Selected cross-seed optimizer | "
                f"kappa={optimized_policy.disagreement_coefficient:g} "
                f"eta={optimized_policy.turnover_regularization:g} "
                f"validation_CAGR={optimized_policy.validation_cagr:.4f}",
                flush=True,
            )

        if not ensemble.run_extended_ablations:
            return _evaluate_baselines_and_finalize(
                data=data,
                experiment=experiment,
                market=market,
                output_dir=output_dir,
                results=results,
                histories=histories,
                experiment_started=experiment_started,
                seed_timings=seed_timings,
                verbose=verbose,
            )

        if verbose:
            print(
                "\n=== Calibrating VNINDEX regime-constrained disagreement ===",
                flush=True,
            )
        regime_policy, regime_grid, regime_calibration = (
            calibrate_regime_disagreement_policy(
                data,
                experiment,
                market,
                ppo,
                ensemble,
                agents,
                policy,
                seed=seeds[0],
                verbose=verbose,
            )
        )
        regime_grid.to_csv(
            output_dir / "regime_disagreement_validation_grid.csv",
            index=False,
        )
        (output_dir / "regime_disagreement_calibration.json").write_text(
            json.dumps(regime_calibration, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        regime_env = make_environment(
            data, experiment, market, split="test", seed=seeds[0]
        )
        daily, metrics = evaluate_regime_disagreement_ensemble(
            regime_env, agents, regime_policy
        )
        name = "PPO_vnindex_regime_overlay_ablation"
        daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
        results[name] = metrics
        histories[name] = daily

        for method, ablation_policy in consensus_ablation_policies(
            optimized_policy, ensemble.optimization_consensus_methods
        ).items():
            ablation_env = make_environment(
                data, experiment, market, split="test", seed=seeds[0]
            )
            ablation_daily, ablation_metrics = (
                evaluate_optimized_disagreement_ensemble(
                    ablation_env, agents, ablation_policy
                )
            )
            ablation_name = f"PPO_consensus_{method}_ablation"
            ablation_daily.to_csv(
                output_dir / f"daily_{ablation_name}.csv", index=False
            )
            results[ablation_name] = ablation_metrics
            histories[ablation_name] = ablation_daily

    baseline_settings = {
        "EqualWeight_daily": 1,
        "EqualWeight_weekly": 5,
        "EqualWeight_monthly": 20,
    }
    for name, interval in baseline_settings.items():
        env = make_environment(data, experiment, market, split="test", seed=42)
        daily, metrics = evaluate_equal_weight(env, rebalance_every=interval)
        daily.to_csv(output_dir / f"daily_{name}.csv", index=False)
        results[name] = metrics
        histories[name] = daily

    buy_hold_env = make_environment(data, experiment, market, split="test", seed=42)
    daily, metrics = evaluate_equal_weight_buy_and_hold(buy_hold_env)
    daily.to_csv(output_dir / "daily_EqualWeight_buy_hold.csv", index=False)
    results["EqualWeight_buy_hold"] = metrics
    histories["EqualWeight_buy_hold"] = daily

    momentum_env = make_environment(data, experiment, market, split="test", seed=42)
    daily, metrics = evaluate_cross_sectional_momentum(
        momentum_env,
        lookback_days=60,
        top_k=15,
        rebalance_every=20,
        cash_weight=0.05,
    )
    momentum_name = "Momentum60_Top15_Monthly"
    daily.to_csv(output_dir / f"daily_{momentum_name}.csv", index=False)
    results[momentum_name] = metrics
    histories[momentum_name] = daily
    inverse_vol_env = make_environment(
        data, experiment, market, split="test", seed=42
    )
    daily, metrics = evaluate_inverse_volatility(
        inverse_vol_env,
        lookback_days=60,
        rebalance_every=20,
        cash_weight=0.05,
    )
    inverse_vol_name = "InverseVolatility60_Monthly"
    daily.to_csv(output_dir / f"daily_{inverse_vol_name}.csv", index=False)
    results[inverse_vol_name] = metrics
    histories[inverse_vol_name] = daily
    results["VNINDEX"] = _vnindex_row(metrics)

    _save_paired_bootstrap_comparisons(
        histories, PROPOSED_STRATEGY, output_dir
    )

    table = metrics_table(results)
    table.to_csv(output_dir / "metrics.csv")
    save_equity_plot(histories, output_dir / "equity_curves.png")
    total_experiment_seconds = time.perf_counter() - experiment_started
    timing_summary = {
        "total_experiment_seconds": total_experiment_seconds,
        "total_experiment_display": format_duration(total_experiment_seconds),
        "seeds": seed_timings,
    }
    (output_dir / "timing_summary.json").write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2), encoding="utf-8"
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
        print("\n=== Out-of-sample metrics ===")
        print(table.reindex(columns=columns).round(4).to_string())
        print(
            f"\nTotal experiment time: "
            f"{format_duration(total_experiment_seconds)} "
            f"({total_experiment_seconds:.1f}s)"
        )
        print(f"\nSaved experiment to: {output_dir.resolve()}")
    return table


def reevaluate_ensemble_from_checkpoints(
    output_path: str | Path,
    *,
    device: str = "auto",
    verbose: bool = True,
) -> pd.DataFrame:
    """Rebuild the compact final strategy suite without retraining PPO."""

    started = time.perf_counter()
    output_dir = Path(output_path)
    config_path = output_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment configuration: {config_path}")
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = configuration.get("backtest_protocol_version")
    if protocol != BACKTEST_PROTOCOL_VERSION:
        raise ValueError(
            "Legacy checkpoints were produced by an audit-unsafe backtest "
            f"protocol ({protocol!r}). Retrain with {BACKTEST_PROTOCOL_VERSION}; "
            "do not reuse the old 2021-2025 results."
        )
    experiment = ExperimentConfig(**configuration["experiment"])
    # The artifact directory is authoritative even if it was moved after training.
    experiment.output_dir = str(output_dir)
    verify_input_fingerprints(configuration, experiment)
    market = MarketConfig(**configuration["market"])
    ppo_data = dict(configuration["ppo"])
    ppo_data["device"] = device
    ppo = PPOConfig(**ppo_data)
    ensemble_data = dict(configuration.get("ensemble", {}))
    current_ensemble = EnsembleConfig()
    for name in (
        "soft_disagreement_k_grid",
        "adaptive_turnover_budget_grid",
        "adaptive_turnover_k_grid",
        "adaptive_cost_penalty_grid",
        "adaptive_cost_k_grid",
        "asset_disagreement_k_grid",
        "hard_partial_rebalance_grid",
        "combined_k_grid",
        "combined_turnover_budget_grid",
        "combined_global_penalty_grid",
        "combined_asset_penalty_grid",
        "combined_consensus_methods",
    ):
        ensemble_data[name] = list(getattr(current_ensemble, name))
    ensemble_data["adaptive_selection_tolerance"] = (
        current_ensemble.adaptive_selection_tolerance
    )
    ensemble_data["minimum_rebalance_grid"] = list(
        current_ensemble.minimum_rebalance_grid
    )
    ensemble_data["reliability_temperature_grid"] = list(
        current_ensemble.reliability_temperature_grid
    )
    ensemble_data["directional_shrinkage_grid"] = list(
        current_ensemble.directional_shrinkage_grid
    )
    ensemble_data["directional_turnover_regularization_grid"] = list(
        current_ensemble.directional_turnover_regularization_grid
    )
    ensemble_data["directional_minimum_validation_improvement"] = (
        current_ensemble.directional_minimum_validation_improvement
    )
    ensemble_data["regime_lookback_days"] = (
        current_ensemble.regime_lookback_days
    )
    ensemble_data["regime_risk_off_exposure_grid"] = list(
        current_ensemble.regime_risk_off_exposure_grid
    )
    ensemble_data["optimization_disagreement_coefficient_grid"] = list(
        current_ensemble.optimization_disagreement_coefficient_grid
    )
    ensemble_data["optimization_turnover_regularization_grid"] = list(
        current_ensemble.optimization_turnover_regularization_grid
    )
    ensemble_data["optimization_primary_consensus_method"] = (
        current_ensemble.optimization_primary_consensus_method
    )
    saved_consensus = ensemble_data.get("optimization_consensus_methods", ())
    ensemble_data["optimization_consensus_methods"] = list(
        dict.fromkeys(
            [
                *saved_consensus,
                *current_ensemble.optimization_consensus_methods,
            ]
        )
    )
    for name in (
        "low_quantiles",
        "high_quantiles",
        "minimum_rebalance_grid",
        "soft_disagreement_k_grid",
        "adaptive_turnover_budget_grid",
        "adaptive_turnover_k_grid",
        "adaptive_cost_penalty_grid",
        "adaptive_cost_k_grid",
        "asset_disagreement_k_grid",
        "hard_partial_rebalance_grid",
        "combined_k_grid",
        "combined_turnover_budget_grid",
        "combined_global_penalty_grid",
        "combined_asset_penalty_grid",
        "combined_consensus_methods",
        "reliability_temperature_grid",
        "directional_shrinkage_grid",
        "directional_turnover_regularization_grid",
        "regime_risk_off_exposure_grid",
        "optimization_disagreement_coefficient_grid",
        "optimization_turnover_regularization_grid",
        "optimization_consensus_methods",
    ):
        if name in ensemble_data:
            ensemble_data[name] = tuple(ensemble_data[name])
    ensemble = EnsembleConfig(**ensemble_data)
    seeds = list(validate_research_seeds(configuration["seeds"]))
    experiment.validate()
    market.validate()
    ppo.validate()
    ensemble.validate()
    device_info = training_device_info(device)
    if verbose:
        print(f"\n{format_device_banner(device_info)}", flush=True)
        print("\n=== Loading trained PPO checkpoints ===", flush=True)
    data = load_market_data(
        experiment.data_path,
        train_start=experiment.train_start,
        train_end=experiment.train_end,
        lookback=experiment.lookback,
        minimum_training_days=experiment.minimum_training_days,
        membership_path=experiment.universe_path,
        require_point_in_time_universe=experiment.require_point_in_time_universe,
        required_end=experiment.test_end,
        expected_universe_size=experiment.expected_universe_size,
    )
    verify_input_fingerprints(configuration, experiment)
    if list(data.tickers) != configuration.get("tickers"):
        raise ValueError("Asset ordering differs from the training artifact")
    agents: list[PPOAgent] = []
    checkpoint_files = configuration.get("checkpoint_files")
    if not isinstance(checkpoint_files, dict):
        raise ValueError(
            "Artifact has no checkpoint fingerprints; retrain with the current "
            "audit protocol"
        )
    for seed in seeds:
        checkpoint = output_dir / f"ppo_seed_{seed}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing PPO checkpoint: {checkpoint}")
        saved_checkpoint = checkpoint_files.get(str(seed))
        if not isinstance(saved_checkpoint, dict) or (
            saved_checkpoint.get("filename") != checkpoint.name
            or saved_checkpoint.get("sha256") != file_sha256(checkpoint)
        ):
            raise ValueError(
                f"Checkpoint fingerprint mismatch for seed {seed}: {checkpoint}"
            )
        agents.append(PPOAgent.load(checkpoint, device=device))
        if verbose:
            print(f"Loaded seed {seed}: {checkpoint.name}", flush=True)

    results: dict[str, dict[str, float]] = {}
    histories: dict[str, pd.DataFrame] = {}
    for agent_seed, agent in zip(seeds, agents, strict=True):
        seed_env = make_environment(
            data, experiment, market, split="test", seed=agent_seed
        )
        seed_daily, seed_metrics = evaluate_agent(seed_env, agent)
        seed_name = f"PPO_seed_{agent_seed}"
        seed_daily.to_csv(output_dir / f"daily_{seed_name}.csv", index=False)
        results[seed_name] = seed_metrics
        histories[seed_name] = seed_daily

    table = _evaluate_final_strategy_suite(
        data=data,
        experiment=experiment,
        market=market,
        ppo=ppo,
        ensemble=ensemble,
        agents=agents,
        seeds=seeds,
        output_dir=output_dir,
        results=results,
        histories=histories,
        started=started,
        timing_key="final_ensemble_reevaluation_seconds",
        verbose=verbose,
    )
    configuration["ensemble"] = dataclass_dict(ensemble)
    configuration["ensemble_evaluation_device"] = device_info
    configuration["proposed_strategy"] = PROPOSED_STRATEGY
    configuration["final_optimized_strategies"] = list(
        FINAL_OPTIMIZED_STRATEGIES
    )
    configuration["consensus_only_strategy"] = CONSENSUS_ONLY_STRATEGY
    configuration["final_compact_report"] = True
    config_path.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return table

    if verbose:
        print(
            "\n=== Calibrating disagreement strategy on disjoint calibration ===",
            flush=True,
        )
    policy, calibration_grid, calibration = calibrate_disagreement_policy(
        data,
        experiment,
        market,
        ppo,
        ensemble,
        agents,
        seed=seeds[0],
        verbose=verbose,
    )
    calibration_grid.to_csv(
        output_dir / "disagreement_validation_grid.csv", index=False
    )
    (output_dir / "disagreement_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calibration_grid.to_csv(
        output_dir / "scalar_gate_validation_grid.csv", index=False
    )
    (output_dir / "scalar_gate_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(
            "Selected disagreement policy | "
            f"q=({policy.low_quantile:.2f},{policy.high_quantile:.2f}) "
            f"thresholds=({policy.low_threshold:.4f},"
            f"{policy.high_threshold:.4f}) "
            f"min_gate={policy.minimum_rebalance:.2f}",
            flush=True,
        )

    if verbose:
        print(
            "\n=== Calibrating primary reliability-weighted strategy ===",
            flush=True,
        )
    (
        primary_policy,
        primary_grid,
        primary_calibration,
    ) = calibrate_reliability_weighted_policy(
        data,
        experiment,
        market,
        ppo,
        ensemble,
        agents,
        policy,
        seeds=seeds,
        verbose=verbose,
    )
    primary_grid.to_csv(
        output_dir / "reliability_weighted_validation_grid.csv", index=False
    )
    primary_grid.to_csv(
        output_dir / "disagreement_validation_grid.csv", index=False
    )
    primary_payload = json.dumps(
        primary_calibration, ensure_ascii=False, indent=2
    )
    (output_dir / "reliability_weighted_calibration.json").write_text(
        primary_payload, encoding="utf-8"
    )
    (output_dir / "disagreement_calibration.json").write_text(
        primary_payload, encoding="utf-8"
    )

    results: dict[str, dict[str, float]] = {}
    for agent_seed, agent in zip(seeds, agents):
        seed_env = make_environment(
            data, experiment, market, split="test", seed=agent_seed
        )
        seed_daily, seed_metrics = evaluate_agent(seed_env, agent)
        seed_name = f"PPO_seed_{agent_seed}"
        seed_daily.to_csv(
            output_dir / f"daily_{seed_name}.csv", index=False
        )
        results[seed_name] = seed_metrics

    mean_env = make_environment(
        data, experiment, market, split="test", seed=seeds[0]
    )
    daily, metrics = evaluate_mean_ensemble(mean_env, agents)
    daily.to_csv(output_dir / "daily_PPO_mean_ensemble.csv", index=False)
    results["PPO_mean_ensemble"] = metrics

    attached_results, _ = evaluate_attached_rebalance_suite(
        data,
        experiment,
        market,
        ppo,
        ensemble,
        agents,
        output_dir,
        seed=seeds[0],
        verbose=verbose,
    )
    results.update(attached_results)

    scalar_gate_env = make_environment(
        data, experiment, market, split="test", seed=seeds[0]
    )
    scalar_daily, scalar_metrics = evaluate_disagreement_ensemble(
        scalar_gate_env, agents, policy
    )
    scalar_name = "PPO_scalar_disagreement_gate_ablation"
    scalar_daily.to_csv(
        output_dir / f"daily_{scalar_name}.csv", index=False
    )
    results[scalar_name] = scalar_metrics

    proposed_env = make_environment(
        data, experiment, market, split="test", seed=seeds[0]
    )
    proposed_daily, proposed_metrics = evaluate_reliability_weighted_ensemble(
        proposed_env, agents, primary_policy
    )
    proposed_daily.to_csv(
        output_dir / f"daily_{RELIABILITY_WEIGHTED_STRATEGY}.csv", index=False
    )
    results[RELIABILITY_WEIGHTED_STRATEGY] = proposed_metrics
    if verbose:
        print(
            "Selected reliability policy | "
            f"temperature={primary_policy.temperature:g} "
            f"weights={np.round(primary_policy.seed_weights, 3).tolist()} "
            f"validation_score={primary_policy.validation_score:.4f}",
            flush=True,
        )

    if verbose:
        print(
            "\n=== Calibrating directional disagreement optimizer ===",
            flush=True,
        )
    (
        directional_policy,
        directional_grid,
        directional_calibration,
    ) = calibrate_directional_disagreement_policy(
        data,
        experiment,
        market,
        ppo,
        ensemble,
        agents,
        policy,
        seed=seeds[0],
        verbose=verbose,
    )
    directional_grid.to_csv(
        output_dir / "directional_disagreement_validation_grid.csv",
        index=False,
    )
    (output_dir / "directional_disagreement_calibration.json").write_text(
        json.dumps(directional_calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    directional_env = make_environment(
        data, experiment, market, split="test", seed=seeds[0]
    )
    directional_daily, directional_metrics = (
        evaluate_directional_disagreement_ensemble(
            directional_env, agents, directional_policy
        )
    )
    directional_name = EQUAL_SEED_DISAGREEMENT_STRATEGY
    directional_daily.to_csv(
        output_dir / f"daily_{directional_name}.csv", index=False
    )
    results[directional_name] = directional_metrics
    if verbose:
        print(
            "Selected equal-seed D3R policy | "
            f"kappa={directional_policy.shrinkage_coefficient:g} "
            f"eta={directional_policy.turnover_regularization:g} "
            f"validation_score={directional_policy.validation_score:.4f}",
            flush=True,
        )

    if directional_only:
        table = _merge_reevaluated_results(
            output_dir,
            results,
            proposed_name=directional_name,
        )
        elapsed = time.perf_counter() - started
        configuration["ensemble"] = dataclass_dict(ensemble)
        configuration["ensemble_evaluation_device"] = device_info
        configuration["directional_only_reevaluation"] = True
        config_path.write_text(
            json.dumps(configuration, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        timing_path = output_dir / "timing_summary.json"
        timing = (
            json.loads(timing_path.read_text(encoding="utf-8"))
            if timing_path.exists()
            else {}
        )
        timing["directional_reevaluation_seconds"] = elapsed
        timing["directional_reevaluation_display"] = format_duration(elapsed)
        timing_path.write_text(
            json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
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
            print("\n=== Directional ensemble out-of-sample metrics ===")
            print(
                table.loc[list(results)]
                .reindex(columns=columns)
                .round(4)
                .to_string()
            )
            print(f"\nDirectional evaluation time: {format_duration(elapsed)}")
            print(f"Saved experiment to: {output_dir.resolve()}")
        return table

    if verbose:
        print(
            "\n=== Calibrating convex cross-seed optimizer ===",
            flush=True,
        )
    (
        optimized_policy,
        optimization_grid,
        optimization_calibration,
    ) = calibrate_optimized_disagreement_policy(
        data,
        experiment,
        market,
        ensemble,
        agents,
        policy,
        seed=seeds[0],
        verbose=verbose,
    )
    optimization_grid.to_csv(
        output_dir / "disagreement_optimization_validation_grid.csv",
        index=False,
    )
    (output_dir / "disagreement_optimization_calibration.json").write_text(
        json.dumps(optimization_calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    optimized_env = make_environment(
        data, experiment, market, split="test", seed=seeds[0]
    )
    optimized_daily, optimized_metrics = evaluate_optimized_disagreement_ensemble(
        optimized_env, agents, optimized_policy
    )
    optimized_name = CROSS_SEED_OPTIMIZER_STRATEGY
    optimized_daily.to_csv(
        output_dir / f"daily_{optimized_name}.csv", index=False
    )
    results[optimized_name] = optimized_metrics
    if verbose:
        print(
            "Selected cross-seed optimizer | "
            f"kappa={optimized_policy.disagreement_coefficient:g} "
            f"eta={optimized_policy.turnover_regularization:g} "
            f"validation_CAGR={optimized_policy.validation_cagr:.4f}",
            flush=True,
        )

    if not extended_ablations:
        table = _merge_reevaluated_results(
            output_dir,
            results,
            proposed_name=PROPOSED_STRATEGY,
        )
        elapsed = time.perf_counter() - started
        configuration["ensemble"] = dataclass_dict(ensemble)
        configuration["ensemble_evaluation_device"] = device_info
        configuration["proposed_strategy"] = PROPOSED_STRATEGY
        configuration["equal_seed_disagreement_strategy"] = (
            EQUAL_SEED_DISAGREEMENT_STRATEGY
        )
        configuration["cross_seed_optimizer_strategy"] = (
            CROSS_SEED_OPTIMIZER_STRATEGY
        )
        configuration["extended_ablations_evaluated"] = False
        config_path.write_text(
            json.dumps(configuration, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        timing_path = output_dir / "timing_summary.json"
        timing = (
            json.loads(timing_path.read_text(encoding="utf-8"))
            if timing_path.exists()
            else {}
        )
        timing["ensemble_reevaluation_seconds"] = elapsed
        timing["ensemble_reevaluation_display"] = format_duration(elapsed)
        timing_path.write_text(
            json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
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
            print("\n=== Core ensemble metrics ===")
            print(
                table.loc[list(results)]
                .reindex(columns=columns)
                .round(4)
                .to_string()
            )
            print(f"\nEnsemble evaluation time: {format_duration(elapsed)}")
            print(f"Saved experiment to: {output_dir.resolve()}")
        return table

    if verbose:
        print(
            "\n=== Calibrating VNINDEX regime-constrained disagreement ===",
            flush=True,
        )
    regime_policy, regime_grid, regime_calibration = (
        calibrate_regime_disagreement_policy(
            data,
            experiment,
            market,
            ppo,
            ensemble,
            agents,
            policy,
            seed=seeds[0],
            verbose=verbose,
        )
    )
    regime_grid.to_csv(
        output_dir / "regime_disagreement_validation_grid.csv", index=False
    )
    (output_dir / "regime_disagreement_calibration.json").write_text(
        json.dumps(regime_calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    regime_env = make_environment(
        data, experiment, market, split="test", seed=seeds[0]
    )
    regime_daily, regime_metrics = evaluate_regime_disagreement_ensemble(
        regime_env, agents, regime_policy
    )
    regime_name = "PPO_vnindex_regime_overlay_ablation"
    regime_daily.to_csv(
        output_dir / f"daily_{regime_name}.csv", index=False
    )
    results[regime_name] = regime_metrics

    for method, ablation_policy in consensus_ablation_policies(
        optimized_policy, ensemble.optimization_consensus_methods
    ).items():
        ablation_env = make_environment(
            data, experiment, market, split="test", seed=seeds[0]
        )
        ablation_daily, ablation_metrics = (
            evaluate_optimized_disagreement_ensemble(
                ablation_env, agents, ablation_policy
            )
        )
        ablation_name = f"PPO_consensus_{method}_ablation"
        ablation_daily.to_csv(
            output_dir / f"daily_{ablation_name}.csv", index=False
        )
        results[ablation_name] = ablation_metrics
    momentum_env = make_environment(
        data, experiment, market, split="test", seed=42
    )
    momentum_daily, momentum_metrics = evaluate_cross_sectional_momentum(
        momentum_env,
        lookback_days=60,
        top_k=15,
        rebalance_every=20,
        cash_weight=0.05,
    )
    momentum_name = "Momentum60_Top15_Monthly"
    momentum_daily.to_csv(
        output_dir / f"daily_{momentum_name}.csv", index=False
    )
    results[momentum_name] = momentum_metrics

    inverse_vol_env = make_environment(
        data, experiment, market, split="test", seed=42
    )
    inverse_vol_daily, inverse_vol_metrics = evaluate_inverse_volatility(
        inverse_vol_env,
        lookback_days=60,
        rebalance_every=20,
        cash_weight=0.05,
    )
    inverse_vol_name = "InverseVolatility60_Monthly"
    inverse_vol_daily.to_csv(
        output_dir / f"daily_{inverse_vol_name}.csv", index=False
    )
    results[inverse_vol_name] = inverse_vol_metrics

    table = _merge_reevaluated_results(
        output_dir,
        results,
        proposed_name=PROPOSED_STRATEGY,
    )

    elapsed = time.perf_counter() - started
    configuration["ensemble"] = dataclass_dict(ensemble)
    configuration["ensemble_evaluation_device"] = device_info
    config_path.write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    timing_path = output_dir / "timing_summary.json"
    timing = (
        json.loads(timing_path.read_text(encoding="utf-8"))
        if timing_path.exists()
        else {}
    )
    timing["ensemble_reevaluation_seconds"] = elapsed
    timing["ensemble_reevaluation_display"] = format_duration(elapsed)
    timing_path.write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
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
        print("\n=== Ensemble out-of-sample metrics ===")
        print(table.loc[list(results)].reindex(columns=columns).round(4).to_string())
        print(f"\nEnsemble evaluation time: {format_duration(elapsed)}")
        print(f"Saved experiment to: {output_dir.resolve()}")
    return table
