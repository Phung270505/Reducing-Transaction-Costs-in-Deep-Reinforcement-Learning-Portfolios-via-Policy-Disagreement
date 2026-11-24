"""Major-revision evaluation suite using frozen PPO checkpoints only.

This command does not train or update a PPO. It adds matched generic execution
controls, paired time-series inference, cross-seed diagnostics, integrity
checks, and publication figures requested by the reviewer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vn30ppo.config import ExperimentConfig, MarketConfig, PPOConfig
from vn30ppo.data import load_market_data
from vn30ppo.evaluation import (
    AdaptiveRebalancePolicy,
    equal_seed_consensus,
    evaluate_adaptive_rebalance_ensemble,
    performance_metrics,
    portfolio_disagreement,
    run_backtest,
)
from vn30ppo.experiment import file_sha256, make_environment, verify_input_fingerprints
from vn30ppo.ppo import PPOAgent, training_device_info
from vn30ppo.research_protocol import validate_research_seeds
from vn30ppo.reviewer_statistics import (
    holm_adjust,
    paired_additive_block_bootstrap,
    paired_metric_block_bootstrap,
    spearman_with_bootstrap,
)


DEFAULT_ARTIFACTS = Path("artifacts/disagreement_ppo_main")
DEFAULT_OUTPUT = Path("artifacts/reviewer_evaluation")
INITIAL_CASH = 1_000_000_000.0


EXISTING_DAILY = {
    "PPO mean": "daily_PPO_mean_ensemble.csv",
    "DAR–Soft": "daily_PPO_soft_disagreement_rebalance.csv",
    "DAR–Combined-v2": "daily_PPO_robust_asset_adaptive_optimizer.csv",
    "DAR–Combined-v1": "daily_PPO_robust_asset_adaptive_optimizer_frozen_v1.csv",
    "Consensus-only": "daily_PPO_consensus_only_gate.csv",
    "Equal Weight": "daily_EqualWeight_monthly.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen PPO checkpoints for the major revision"
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-replications", type=int, default=20_000)
    return parser.parse_args()


def load_frozen_context(artifact_dir: Path, device: str):
    configuration = json.loads(
        (artifact_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    experiment = ExperimentConfig(**configuration["experiment"])
    experiment.output_dir = str(artifact_dir)
    market = MarketConfig(**configuration["market"])
    ppo_data = dict(configuration["ppo"])
    ppo_data["device"] = device
    ppo = PPOConfig(**ppo_data)
    seeds = list(validate_research_seeds(configuration["seeds"]))
    verify_input_fingerprints(configuration, experiment)
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
    if list(data.tickers) != configuration["tickers"]:
        raise ValueError("ticker order differs from the frozen training artifact")
    agents = []
    for seed in seeds:
        checkpoint = artifact_dir / f"ppo_seed_{seed}.pt"
        fingerprint = configuration["checkpoint_files"][str(seed)]
        if fingerprint["filename"] != checkpoint.name or fingerprint["sha256"] != file_sha256(checkpoint):
            raise ValueError(f"checkpoint fingerprint mismatch: {checkpoint}")
        agents.append(PPOAgent.load(checkpoint, device=device))
    return configuration, experiment, market, ppo, data, seeds, agents


def evaluate_policy(data, experiment, market, agents, seed: int, split: str, policy):
    env = make_environment(data, experiment, market, split=split, seed=seed)
    return evaluate_adaptive_rebalance_ensemble(env, agents, policy)


def choose_matched_controls(
    artifact_dir: Path,
    output_dir: Path,
    data,
    experiment,
    market,
    agents,
    seed: int,
) -> tuple[dict[str, AdaptiveRebalancePolicy], pd.DataFrame]:
    grid = pd.read_csv(artifact_dir / "adaptive_rebalance_validation_grid.csv")
    selected_soft = grid[(grid["method"] == "soft") & grid["selected"]]
    if len(selected_soft) != 1:
        raise ValueError("expected exactly one calibration-selected DAR–Soft")
    target_turnover = float(selected_soft.iloc[0]["annual_turnover"])

    candidates: list[tuple[str, float, AdaptiveRebalancePolicy]] = []
    alphas = sorted(set(np.round(np.r_[np.arange(0.50, 1.001, 0.05), np.arange(0.85, 1.001, 0.01)], 2)))
    for alpha in alphas:
        candidates.append(
            ("fixed_alpha", float(alpha), AdaptiveRebalancePolicy("fixed_alpha", fixed_alpha=float(alpha)))
        )
    for budget in np.arange(0.10, 1.001, 0.10):
        candidates.append(
            (
                "fixed_budget",
                float(budget),
                AdaptiveRebalancePolicy(
                    "turnover_constraint",
                    disagreement_k=0.0,
                    base_turnover_budget=float(budget),
                ),
            )
        )
    for penalty in (0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08):
        candidates.append(
            (
                "fixed_l1_penalty",
                float(penalty),
                AdaptiveRebalancePolicy(
                    "cost_penalty",
                    disagreement_k=0.0,
                    base_cost_penalty=float(penalty),
                ),
            )
        )

    rows = []
    policy_lookup: dict[tuple[str, float], AdaptiveRebalancePolicy] = {}
    for family, parameter, policy in candidates:
        _, metrics = evaluate_policy(
            data, experiment, market, agents, seed, "calibration", policy
        )
        rows.append(
            {
                "family": family,
                "parameter": parameter,
                "target_dar_soft_annual_turnover": target_turnover,
                "annual_turnover": metrics["annual_turnover"],
                "turnover_gap": abs(metrics["annual_turnover"] - target_turnover),
                "cagr": metrics["cagr"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "total_transaction_cost": metrics["total_transaction_cost"],
            }
        )
        policy_lookup[(family, parameter)] = policy
    calibration = pd.DataFrame(rows)
    calibration["selected"] = False
    selected: dict[str, AdaptiveRebalancePolicy] = {}
    display_names = {
        "fixed_alpha": "Fixed-alpha matched",
        "fixed_budget": "Fixed-budget matched",
        "fixed_l1_penalty": "Fixed-L1 matched",
    }
    for family, group in calibration.groupby("family", sort=False):
        winner_index = group.sort_values(["turnover_gap", "parameter"]).index[0]
        calibration.loc[winner_index, "selected"] = True
        parameter = float(calibration.loc[winner_index, "parameter"])
        selected[display_names[family]] = policy_lookup[(family, parameter)]
    calibration.to_csv(output_dir / "matched_control_calibration.csv", index=False)
    return selected, calibration


def evaluate_shuffled_gate(
    env,
    agents,
    gates: np.ndarray,
    *,
    permutation_seed: int = 20_260_828,
):
    gates = np.asarray(gates, dtype=np.float64)
    if gates.ndim != 1 or not len(gates):
        raise ValueError("a non-empty one-dimensional gate schedule is required")
    rng = np.random.default_rng(permutation_seed)
    shuffled = gates.copy()
    if len(shuffled) > 1:
        shuffled[1:] = rng.permutation(shuffled[1:])
    decision_number = 0
    interval = int(agents[0].config.rebalance_interval)

    def policy(observation, current_env, step):
        nonlocal decision_number
        if step % interval:
            return None
        proposals = np.stack(
            [current_env.action_to_weights(agent.predict(observation)) for agent in agents]
        )
        consensus = equal_seed_consensus(proposals, "mean")
        current = current_env.current_hold_weights()
        gate = 1.0 if step == 0 else float(shuffled[min(decision_number, len(shuffled) - 1)])
        target = current + gate * (consensus - current)
        diagnostics = portfolio_disagreement(proposals)
        consensus_change = float(0.5 * np.abs(consensus - current).sum())
        diagnostics.update(
            {
                "rebalance_gate": gate,
                "consensus_requested_change": consensus_change,
                "adaptive_requested_change": gate * consensus_change,
                "disagreement_turnover_reduction": (1.0 - gate) * consensus_change,
                "maximum_seed_weight": 1.0 / len(agents),
                "effective_seed_count": float(len(agents)),
            }
        )
        decision_number += 1
        return current_env.weights_to_action(target), diagnostics

    daily = run_backtest(env, policy)
    return daily, performance_metrics(daily, initial_cash=env.config.initial_cash)


def run_generic_controls(
    artifact_dir: Path,
    output_dir: Path,
    data,
    experiment,
    market,
    agents,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    selected, calibration = choose_matched_controls(
        artifact_dir, output_dir, data, experiment, market, agents, seed
    )
    selected["Geometric-median only"] = AdaptiveRebalancePolicy(
        "consensus", consensus_method="geometric_median"
    )
    histories: dict[str, pd.DataFrame] = {}
    metric_rows = []
    for name, policy in selected.items():
        daily, metrics = evaluate_policy(
            data, experiment, market, agents, seed, "test", policy
        )
        histories[name] = daily
        daily.to_csv(output_dir / f"daily_{name.replace(' ', '_')}.csv", index=False)
        metric_rows.append({"strategy": name, **metrics})

    soft = pd.read_csv(
        artifact_dir / EXISTING_DAILY["DAR–Soft"], parse_dates=["date"]
    )
    gates = soft.loc[soft["disagreement_to_consensus_tv"].notna(), "rebalance_gate"].to_numpy()
    shuffled_env = make_environment(data, experiment, market, split="test", seed=seed)
    shuffled_daily, shuffled_metrics = evaluate_shuffled_gate(
        shuffled_env, agents, gates
    )
    name = "Shuffled-disagreement timing"
    histories[name] = shuffled_daily
    shuffled_daily.to_csv(output_dir / "daily_Shuffled_disagreement_timing.csv", index=False)
    metric_rows.append({"strategy": name, **shuffled_metrics})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "generic_control_metrics.csv", index=False)
    return histories, metrics, calibration


def load_all_histories(
    artifact_dir: Path, control_histories: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    histories = {
        name: pd.read_csv(artifact_dir / filename, parse_dates=["date"])
        for name, filename in EXISTING_DAILY.items()
    }
    histories.update(control_histories)
    return histories


def statistical_comparisons(
    histories: dict[str, pd.DataFrame],
    output_dir: Path,
    replications: int,
) -> pd.DataFrame:
    pairs = [
        ("DAR–Soft", "PPO mean", "primary"),
        ("DAR–Combined-v2", "PPO mean", "primary"),
        ("DAR–Combined-v1", "PPO mean", "secondary-development"),
        ("DAR–Soft", "Fixed-alpha matched", "matched-control"),
        ("DAR–Soft", "Fixed-budget matched", "matched-control"),
        ("DAR–Soft", "Fixed-L1 matched", "matched-control"),
        ("DAR–Soft", "Shuffled-disagreement timing", "timing-control"),
    ]
    rows = []
    for block_length in (10, 20, 40):
        for strategy_name, baseline_name, scope in pairs:
            strategy = histories[strategy_name]
            baseline = histories[baseline_name]
            if not strategy["date"].equals(baseline["date"]):
                raise ValueError(f"unaligned daily histories: {strategy_name}, {baseline_name}")
            result = paired_metric_block_bootstrap(
                strategy["daily_return"].to_numpy(),
                baseline["daily_return"].to_numpy(),
                strategy["date"],
                block_length=block_length,
                replications=replications,
                seed=20_260_828 + block_length,
            )
            result.insert(0, "scope", scope)
            result.insert(0, "baseline", baseline_name)
            result.insert(0, "strategy", strategy_name)
            rows.extend(result.to_dict("records"))
            for metric, columns in {
                "total_turnover": ["turnover"],
                "transaction_cost_vnd": ["broker_fees", "sell_tax", "slippage_cost"],
            }.items():
                left = strategy[columns].sum(axis=1).to_numpy()
                right = baseline[columns].sum(axis=1).to_numpy()
                additive = paired_additive_block_bootstrap(
                    left,
                    right,
                    metric=metric,
                    block_length=block_length,
                    replications=replications,
                    seed=20_260_828 + block_length,
                )
                rows.append(
                    {
                        "strategy": strategy_name,
                        "baseline": baseline_name,
                        "scope": scope,
                        **additive,
                        "elapsed_calendar_days": float("nan"),
                        "elapsed_years": float("nan"),
                    }
                )
    report = pd.DataFrame(rows)
    report["holm_adjusted_p"] = np.nan
    for _, group in report.groupby(["block_length", "metric"], dropna=False):
        report.loc[group.index, "holm_adjusted_p"] = holm_adjust(
            group["two_sided_p_value"]
        )
    report.to_csv(output_dir / "paired_metric_bootstrap.csv", index=False)
    return report


def annualization_audit(histories: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    rows = []
    for name, daily in histories.items():
        metrics = performance_metrics(daily, initial_cash=INITIAL_CASH)
        rows.append(
            {
                "strategy": name,
                "observations": len(daily),
                "first_date": daily["date"].min().date().isoformat(),
                "last_date": daily["date"].max().date().isoformat(),
                "elapsed_calendar_days": metrics["cagr_elapsed_calendar_days"],
                "elapsed_years": metrics["cagr_elapsed_years"],
                "total_return": metrics["total_return"],
                "calendar_cagr": metrics["cagr"],
                "annual_volatility_252": metrics["annual_volatility"],
                "zero_rate_sharpe_252": metrics["sharpe_ratio"],
                "annual_turnover_252": metrics["annual_turnover"],
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "annualization_audit.csv", index=False)
    return report


def disagreement_diagnostics(
    artifact_dir: Path,
    histories: dict[str, pd.DataFrame],
    output_dir: Path,
    replications: int,
) -> dict[str, object]:
    seeds = [5, 10, 15, 20, 25]
    seed_frames = [
        pd.read_csv(artifact_dir / f"daily_PPO_seed_{seed}.csv", parse_dates=["date"])
        for seed in seeds
    ]
    mean = histories["PPO mean"].reset_index(drop=True)
    target_columns = [c for c in mean.columns if c.startswith("target_")]
    review_indices = np.flatnonzero(mean["disagreement_to_consensus_tv"].notna().to_numpy())
    pairwise_sum = np.zeros((len(seeds), len(seeds)))
    decision_rows = []
    loo_values = []
    for row_index in review_indices:
        proposals = np.stack(
            [frame.loc[row_index, target_columns].to_numpy(dtype=float) for frame in seed_frames]
        )
        consensus = proposals.mean(axis=0)
        diagnostics = portfolio_disagreement(proposals)
        for i in range(len(seeds)):
            for j in range(len(seeds)):
                pairwise_sum[i, j] += 0.5 * np.abs(proposals[i] - proposals[j]).sum()
            leave_one_out = np.delete(proposals, i, axis=0).mean(axis=0)
            loo_values.append(
                {
                    "date": mean.loc[row_index, "date"],
                    "omitted_seed": seeds[i],
                    "tv_from_full_consensus": 0.5 * np.abs(leave_one_out - consensus).sum(),
                }
            )
        end = row_index + 20
        forward_return = (
            np.prod(1.0 + mean.loc[row_index : end - 1, "daily_return"].to_numpy()) - 1.0
            if end <= len(mean)
            else float("nan")
        )
        decision_rows.append(
            {
                "date": mean.loc[row_index, "date"],
                "window_end": mean.loc[end - 1, "date"] if end <= len(mean) else pd.NaT,
                "forward_20_session_net_return_ppo_mean": forward_return,
                **diagnostics,
                "cash_weight_std": float(proposals[:, -1].std(ddof=0)),
            }
        )
    decisions = pd.DataFrame(decision_rows)
    loo = pd.DataFrame(loo_values)
    matrix = pd.DataFrame(
        pairwise_sum / len(review_indices),
        index=[f"seed_{seed}" for seed in seeds],
        columns=[f"seed_{seed}" for seed in seeds],
    )
    decisions.to_csv(output_dir / "decision_disagreement_diagnostics.csv", index=False)
    loo.to_csv(output_dir / "leave_one_seed_out_consensus.csv", index=False)
    matrix.to_csv(output_dir / "pairwise_seed_tv_matrix.csv")
    correlations = {
        "D_to_consensus_vs_pairwise": spearman_with_bootstrap(
            decisions["disagreement_to_consensus_tv"].to_numpy(),
            decisions["disagreement_mean_pairwise_tv"].to_numpy(),
            replications=replications,
        ),
        "D_to_consensus_vs_forward_return": spearman_with_bootstrap(
            decisions["disagreement_to_consensus_tv"].to_numpy(),
            decisions["forward_20_session_net_return_ppo_mean"].to_numpy(),
            replications=replications,
        ),
        "D_pairwise_vs_forward_return": spearman_with_bootstrap(
            decisions["disagreement_mean_pairwise_tv"].to_numpy(),
            decisions["forward_20_session_net_return_ppo_mean"].to_numpy(),
            replications=replications,
        ),
    }
    (output_dir / "disagreement_correlations.json").write_text(
        json.dumps(correlations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"decisions": decisions, "loo": loo, "matrix": matrix, "correlations": correlations}


def seed_and_unseen_asset_audit(artifact_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(artifact_dir / "metrics.csv")
    seed_metrics = metrics[metrics["strategy"].str.startswith("PPO_seed_")].copy()
    rows = []
    for metric in ("total_return", "cagr", "sharpe_ratio", "maximum_drawdown", "annual_turnover"):
        values = seed_metrics[metric].to_numpy(dtype=float)
        rows.append(
            {
                "metric": metric,
                "n_seeds": len(values),
                "mean": values.mean(),
                "sample_sd": values.std(ddof=1),
                "median": np.median(values),
                "q25": np.quantile(values, 0.25),
                "q75": np.quantile(values, 0.75),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )
    seed_summary = pd.DataFrame(rows)
    seed_summary.to_csv(output_dir / "seed_uncertainty_summary.csv", index=False)

    unseen = ["SSB", "TCX", "VPL"]
    exposure_rows = []
    for seed in (5, 10, 15, 20, 25):
        daily = pd.read_csv(artifact_dir / f"daily_PPO_seed_{seed}.csv", parse_dates=["date"])
        decisions = daily[daily["disagreement_to_consensus_tv"].notna()] if "disagreement_to_consensus_tv" in daily else daily.iloc[::20]
        for ticker in unseen:
            exposure_rows.append(
                {
                    "strategy": f"PPO seed {seed}",
                    "ticker": ticker,
                    "mean_target_on_decision_days": decisions[f"target_{ticker}"].mean(),
                    "max_target_on_decision_days": decisions[f"target_{ticker}"].max(),
                    "decision_days_positive_target": int((decisions[f"target_{ticker}"] > 1e-8).sum()),
                    "mean_realized_weight": daily[f"weight_{ticker}"].mean(),
                    "max_realized_weight": daily[f"weight_{ticker}"].max(),
                }
            )
    unseen_report = pd.DataFrame(exposure_rows)
    unseen_report.to_csv(output_dir / "unseen_asset_exposure.csv", index=False)
    return seed_summary, unseen_report


def combined_budget_and_cost_audit(
    histories: dict[str, pd.DataFrame], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    budget_rows = []
    for name in ("DAR–Combined-v1", "DAR–Combined-v2"):
        daily = histories[name]
        review = daily[daily["disagreement_to_consensus_tv"].notna()].copy()
        budget = review["adaptive_turnover_budget"].to_numpy(dtype=float)
        adaptive = review["adaptive_requested_change"].to_numpy(dtype=float)
        consensus = review["consensus_requested_change"].to_numpy(dtype=float)
        budget_rows.append(
            {
                "strategy": name,
                "decision_days": len(review),
                "budget_min": budget.min(),
                "budget_median": np.median(budget),
                "budget_max": budget.max(),
                "adaptive_change_max": adaptive.max(),
                "budget_saturation_count": int(np.isclose(adaptive, budget, atol=1e-7).sum()),
                "consensus_change_exceeds_budget_count": int((consensus > budget + 1e-9).sum()),
                "mean_target_turnover_reduction": float((consensus - adaptive).mean()),
            }
        )
    budget_report = pd.DataFrame(budget_rows)
    budget_report.to_csv(output_dir / "combined_budget_diagnostics.csv", index=False)

    mean = histories["PPO mean"]
    soft = histories["DAR–Soft"]
    mean_cost = float(mean[["broker_fees", "sell_tax", "slippage_cost"]].sum().sum())
    soft_cost = float(soft[["broker_fees", "sell_tax", "slippage_cost"]].sum().sum())
    absolute = mean_cost - soft_cost
    elapsed_years = (soft["date"].iloc[-1] - soft["date"].iloc[0]).days / 365.25
    cost_report = pd.DataFrame(
        [
            {
                "comparison": "DAR–Soft vs PPO mean",
                "ppo_mean_cost_vnd": mean_cost,
                "dar_soft_cost_vnd": soft_cost,
                "absolute_saving_vnd": absolute,
                "relative_cost_reduction": absolute / mean_cost,
                "saving_fraction_initial_nav": absolute / INITIAL_CASH,
                "saving_basis_points_initial_nav_total": absolute / INITIAL_CASH * 10_000,
                "saving_basis_points_initial_nav_per_year": absolute / INITIAL_CASH * 10_000 / elapsed_years,
                "note": "Economic cost total only; gross-return path requires a full friction rerun and is not inferred.",
            }
        ]
    )
    cost_report.to_csv(output_dir / "cost_economic_significance.csv", index=False)
    return budget_report, cost_report


def integrity_audit(configuration: dict, output_dir: Path) -> pd.DataFrame:
    experiment = configuration["experiment"]
    rows = [
        {
            "check": "Point-in-time membership",
            "status": "FAIL/SCOPE LIMITATION" if not experiment.get("require_point_in_time_universe", False) else "PASS",
            "evidence": f"require_point_in_time_universe={experiment.get('require_point_in_time_universe')}",
        },
        {
            "check": "Feature normalization fitted on train only",
            "status": "PASS (CODE)",
            "evidence": "vn30ppo/data.py load_market_data and training-only finite mean/std",
        },
        {
            "check": "Open-time volume cap",
            "status": "PASS (CODE+TEST)",
            "evidence": "environment.py uses volumes[index-1]; tests/test_environment.py verifies same-day volume is ignored",
        },
        {
            "check": "Historical HOSE lot size",
            "status": "PASS (CODE+TEST)",
            "evidence": "lot 10 before 2021-01-04, lot 100 afterwards; odd-lot date separately configured",
        },
        {
            "check": "Corporate-action accounting and raw execution prices",
            "status": "NOT VERIFIED",
            "evidence": "Vendor OHLC is described as adjusted; no separate dividend/split cash-and-holdings ledger was found",
        },
        {
            "check": "Untouched confirmatory holdout",
            "status": "FAIL/SCOPE LIMITATION",
            "evidence": "2024-2025 informed development and is relabeled retrospective development evaluation",
        },
        {
            "check": "Checkpoint integrity",
            "status": "PASS",
            "evidence": "Every loaded checkpoint is checked against SHA-256 in experiment_config.json",
        },
    ]
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "leakage_integrity_audit.csv", index=False)
    return report


def create_figures(
    histories: dict[str, pd.DataFrame],
    diagnostics: dict[str, object],
    artifact_dir: Path,
    output_dir: Path,
) -> None:
    # Original Matplotlib defaults were 10 pt for body/labels, 12 pt for
    # titles, and 10 pt for legends/ticks.  The IEEE revision requests a 30%
    # increase, rounded to integer point sizes.
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.titlesize": 16,
        }
    )
    figure_dir = output_dir / "fig"
    figure_dir.mkdir(parents=True, exist_ok=True)
    decisions: pd.DataFrame = diagnostics["decisions"]
    matrix: pd.DataFrame = diagnostics["matrix"]
    loo: pd.DataFrame = diagnostics["loo"]

    fig, ax = plt.subplots(figsize=(12, 6.3), constrained_layout=True)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, width, height, text, color, fontsize=11):
        patch = plt.Rectangle(
            (x, y), width, height, facecolor=color, edgecolor="#30475e", linewidth=1.1
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, style="-"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "lw": 1.1, "linestyle": style, "color": "#30475e"},
        )

    box(0.2, 5.35, 2.0, 0.85, "Stock sequence\n30 × 20 × 12", "#dceefa")
    box(0.2, 4.15, 2.0, 0.85, "VN-Index sequence\n20 × 11", "#dceefa")
    box(0.2, 2.95, 2.0, 0.85, "Asset state\n30 × 4: mask, available\nshares, two T+2 buckets", "#e7f4e4", 10)
    box(0.2, 1.75, 2.0, 0.85, "Global state\n7: cash, pending cash,\ndrawdown, NAV, TO, cost", "#e7f4e4", 10)
    box(2.75, 5.05, 1.65, 1.05, "Shared GRU\n+ temporal attention", "#fff1c9")
    box(2.75, 3.65, 1.65, 1.05, "Market GRU\n+ temporal attention", "#fff1c9")
    box(2.75, 2.15, 1.65, 1.05, "Portfolio/context\nencoder", "#fff1c9")
    box(4.95, 4.05, 1.85, 1.25, "Context fusion\n2-layer Transformer\n4 heads + mask", "#eadcf8")
    box(7.35, 4.75, 1.55, 0.9, "Actor mean\n31 latent logits", "#fbdad5")
    box(7.35, 3.35, 1.55, 0.9, "Critic V(s)", "#fbdad5")
    box(9.35, 5.15, 2.15, 0.9, "Training: diagonal Normal\nlog-prob on latent logits", "#f9e0e0", 10)
    box(9.35, 3.75, 2.15, 0.9, "Inference: mean logits\n+ momentum prior", "#f9e0e0", 10)
    box(9.35, 2.25, 2.15, 0.9, "Masked softmax\n→ target weights", "#d9f0e5")
    box(9.35, 0.75, 2.15, 0.9, "Execution: Open, lot,\nT+2, fees, liquidity\n→ realized weights/NAV", "#d9f0e5", 10)
    for y1, y2 in ((5.78, 5.58), (4.58, 4.18), (3.38, 2.68), (2.18, 2.68)):
        arrow(2.2, y1, 2.75, y2)
    arrow(4.4, 5.58, 4.95, 4.85)
    arrow(4.4, 4.18, 4.95, 4.55)
    arrow(4.4, 2.68, 4.95, 4.25)
    arrow(6.8, 4.8, 7.35, 5.2)
    arrow(6.8, 4.45, 7.35, 3.8)
    arrow(8.9, 5.2, 9.35, 5.6, "--")
    arrow(8.9, 5.2, 9.35, 4.2)
    arrow(10.42, 3.75, 10.42, 3.15)
    arrow(10.42, 2.25, 10.42, 1.65)
    ax.text(6.0, 6.55, "Base PPO Architecture and Execution Path", ha="center", fontsize=17, weight="bold")
    ax.text(9.45, 6.35, "training path", fontsize=10, color="#555555")
    ax.text(9.45, 3.48, "deterministic inference path", fontsize=10, color="#555555")
    fig.savefig(figure_dir / "architecture_complete.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    axes[0, 0].hist(decisions["disagreement_to_consensus_tv"], bins=9, alpha=0.75, label="$D_c$")
    axes[0, 0].hist(decisions["disagreement_mean_pairwise_tv"], bins=9, alpha=0.60, label="$D_p$")
    axes[0, 0].set_title("Decision-Date Disagreement")
    axes[0, 0].legend()
    axes[0, 1].plot(decisions["date"], decisions["disagreement_to_consensus_tv"], marker="o", ms=3, label="$D_c$")
    axes[0, 1].plot(decisions["date"], decisions["disagreement_mean_pairwise_tv"], marker="s", ms=3, label="$D_p$")
    axes[0, 1].set_title("Disagreement over Time")
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[0, 1].legend()
    image = axes[1, 0].imshow(matrix.to_numpy(), cmap="Blues", vmin=0)
    axes[1, 0].set_xticks(range(len(matrix)), matrix.columns, rotation=30)
    axes[1, 0].set_yticks(range(len(matrix)), matrix.index)
    axes[1, 0].set_title("Pairwise Total Variation by Seed")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    loo.boxplot(column="tv_from_full_consensus", by="omitted_seed", ax=axes[1, 1], grid=False)
    axes[1, 1].set_title("Leave-One-Seed-Out Sensitivity")
    axes[1, 1].set_xlabel("Omitted seed")
    axes[1, 1].set_ylabel("TV from five-seed consensus")
    fig.suptitle("")
    fig.savefig(figure_dir / "disagreement_diagnostics.png", dpi=220)
    plt.close(fig)

    usable = decisions.dropna(subset=["forward_20_session_net_return_ppo_mean"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.scatter(
        usable["disagreement_to_consensus_tv"],
        100 * usable["forward_20_session_net_return_ppo_mean"],
        s=38,
        alpha=0.8,
    )
    if len(usable) >= 2:
        slope, intercept = np.polyfit(
            usable["disagreement_to_consensus_tv"],
            usable["forward_20_session_net_return_ppo_mean"],
            1,
        )
        x = np.linspace(usable["disagreement_to_consensus_tv"].min(), usable["disagreement_to_consensus_tv"].max(), 100)
        ax.plot(x, 100 * (intercept + slope * x), color="tab:red", lw=1.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("$D_c$ at decision date")
    ax.set_ylabel("20-session forward PPO mean net return (%)")
    ax.set_title("Disagreement vs. Subsequent Return")
    fig.savefig(figure_dir / "disagreement_forward_return_scatter.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(10.5, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(2.0, 1.0), width_ratios=(2.0, 1.25))
    ax = fig.add_subplot(grid[0, :])
    seed_navs = []
    for seed in (5, 10, 15, 20, 25):
        daily = pd.read_csv(artifact_dir / f"daily_PPO_seed_{seed}.csv", parse_dates=["date"])
        normalized = daily["nav"].to_numpy(dtype=float) / INITIAL_CASH
        seed_navs.append(normalized)
        ax.plot(daily["date"], normalized, lw=1.0, alpha=0.35, color="gray")
    seed_navs_array = np.vstack(seed_navs)
    ax.fill_between(
        daily["date"],
        seed_navs_array.min(axis=0),
        seed_navs_array.max(axis=0),
        color="gray",
        alpha=0.10,
        label="Range across five PPO seeds",
    )
    styles = (
        ("PPO mean", "black", 2.6, "-"),
        ("DAR–Soft", "tab:blue", 2.2, "--"),
        ("DAR–Combined-v2", "tab:green", 2.0, "-."),
        ("Fixed-alpha matched", "tab:orange", 2.0, ":"),
        ("Shuffled-disagreement timing", "tab:red", 1.8, (0, (3, 1, 1, 1))),
    )
    for name, color, width, linestyle in styles:
        daily = histories[name]
        ax.plot(
            daily["date"],
            daily["nav"] / INITIAL_CASH,
            label=name,
            color=color,
            lw=width,
            linestyle=linestyle,
        )
    ax.set_ylabel("NAV / initial capital")
    ax.set_title("(a) Normalized NAV Paths")
    ax.legend(ncol=3, fontsize=10, loc="upper left")

    ax_difference = fig.add_subplot(grid[1, 0])
    mean_nav = histories["PPO mean"].set_index("date")["nav"]
    for name, color, width, linestyle in styles[1:]:
        strategy_nav = histories[name].set_index("date")["nav"].reindex(mean_nav.index)
        relative_basis_points = 10_000 * (strategy_nav / mean_nav - 1.0)
        ax_difference.plot(
            relative_basis_points.index,
            relative_basis_points,
            label=name,
            color=color,
            lw=width,
            linestyle=linestyle,
        )
    ax_difference.axhline(0, color="black", lw=1.0)
    ax_difference.set_title("(b) NAV Difference from PPO Mean")
    ax_difference.set_ylabel("Relative NAV (bps)")
    ax_difference.set_xlabel("Date")
    ax_difference.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax_difference.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    main_metrics = pd.read_csv(artifact_dir / "metrics.csv").set_index("strategy")
    control_metrics = pd.read_csv(output_dir / "generic_control_metrics.csv").set_index("strategy")
    metric_rows = {
        "PPO mean": main_metrics.loc["PPO_mean_ensemble"],
        "DAR–Soft": main_metrics.loc["PPO_soft_disagreement_rebalance"],
        "Combined-v2": main_metrics.loc["PPO_robust_asset_adaptive_optimizer"],
        "Fixed-alpha": control_metrics.loc["Fixed-alpha matched"],
        "Shuffled": control_metrics.loc["Shuffled-disagreement timing"],
    }
    mean_turnover = float(metric_rows["PPO mean"]["annual_turnover"])
    turnover_changes = np.array(
        [100 * (float(row["annual_turnover"]) / mean_turnover - 1.0) for row in metric_rows.values()]
    )
    ax_turnover = fig.add_subplot(grid[1, 1])
    bar_colors = ["black", "tab:blue", "tab:green", "tab:orange", "tab:red"]
    positions = np.arange(len(metric_rows))
    ax_turnover.barh(positions, turnover_changes, color=bar_colors, alpha=0.82)
    short_names = ["PPO mean", "DAR–Soft", "Combined-v2", "Fixed-α", "Shuffled"]
    ax_turnover.set_yticks(positions, short_names, fontsize=10)
    ax_turnover.invert_yaxis()
    ax_turnover.axvline(0, color="black", lw=1.0)
    ax_turnover.set_xlabel("Change (%)")
    cost_summary = pd.read_csv(output_dir / "cost_economic_significance.csv").iloc[0]
    cost_reduction = 100 * float(cost_summary["relative_cost_reduction"])
    ax_turnover.set_xlim(-7.2, 1.0)
    ax_turnover.set_title(
        f"(c) Annual Turnover Change\nDAR–Soft cost reduction: {cost_reduction:.2f}%"
    )
    for position, value in zip(positions, turnover_changes):
        if value < -1.0:
            x_position, alignment, text_color = value / 2.0, "center", "white"
        elif value < -0.05:
            x_position, alignment, text_color = value - 0.12, "right", "black"
        else:
            x_position, alignment, text_color = 0.12, "left", "black"
        ax_turnover.text(
            x_position,
            position,
            f"{value:+.1f}%",
            va="center",
            ha=alignment,
            fontsize=10,
            color=text_color,
            weight="bold",
        )

    fig.suptitle("Development-Period Performance and Trading-Intensity Comparison", weight="bold")
    fig.savefig(figure_dir / "nav_all_seeds_and_controls.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(training_device_info(args.device))
    configuration, experiment, market, _, data, seeds, agents = load_frozen_context(
        args.artifacts, args.device
    )
    print("Loaded frozen PPO checkpoints; no training will be performed.")
    controls, control_metrics, calibration = run_generic_controls(
        args.artifacts,
        args.output,
        data,
        experiment,
        market,
        agents,
        seeds[0],
    )
    histories = load_all_histories(args.artifacts, controls)
    comparisons = statistical_comparisons(
        histories, args.output, args.bootstrap_replications
    )
    annualization = annualization_audit(histories, args.output)
    diagnostics = disagreement_diagnostics(
        args.artifacts, histories, args.output, args.bootstrap_replications
    )
    seed_summary, unseen = seed_and_unseen_asset_audit(args.artifacts, args.output)
    budget, cost = combined_budget_and_cost_audit(histories, args.output)
    integrity = integrity_audit(configuration, args.output)
    create_figures(histories, diagnostics, args.artifacts, args.output)
    manifest = {
        "training_performed": False,
        "checkpoint_sha256_verified": True,
        "development_period": [experiment.test_start, experiment.test_end],
        "bootstrap_replications": args.bootstrap_replications,
        "block_lengths": [10, 20, 40],
        "generic_controls": control_metrics["strategy"].tolist(),
        "outputs": sorted(path.name for path in args.output.iterdir()),
        "scope_limitations": [
            "static ex-post equity basket; no point-in-time VN30 membership",
            "2024-2025 is a retrospective development evaluation, not an untouched confirmatory test",
            "corporate-action accounting is not independently verified",
            "no PPO architecture or reward ablation because retraining was excluded",
        ],
    }
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved reviewer analysis to {args.output.resolve()}")
    print(control_metrics[["strategy", "cagr", "sharpe_ratio", "annual_turnover"]].to_string(index=False))
    print(f"paired rows={len(comparisons)}, decisions={len(diagnostics['decisions'])}, annualization rows={len(annualization)}")
    print(f"seed summaries={len(seed_summary)}, unseen rows={len(unseen)}, budget rows={len(budget)}, integrity checks={len(integrity)}, cost rows={len(cost)}, calibration candidates={len(calibration)}")


if __name__ == "__main__":
    main()
