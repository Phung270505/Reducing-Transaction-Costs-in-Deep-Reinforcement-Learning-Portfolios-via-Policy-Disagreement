from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vn30ppo.evaluation import (
    AdaptiveRebalancePolicy,
    DirectionalDisagreementPolicy,
    DisagreementPolicy,
    DisagreementOptimizationPolicy,
    cap_stock_exposure,
    adaptive_rebalance_target,
    centered_l1_simplex_prox,
    disagreement_controlled_target,
    disagreement_rebalance_gate,
    equal_seed_consensus,
    evaluate_cross_sectional_momentum,
    evaluate_equal_weight,
    evaluate_inverse_volatility,
    paired_circular_block_bootstrap,
    paired_hac_mean_test,
    performance_metrics,
    portfolio_disagreement,
    optimize_from_seed_disagreement,
    optimize_directional_disagreement,
    project_with_turnover_budget,
)
from tests.test_environment import make_env
from vn30ppo.config import PPOConfig
from vn30ppo.experiment import (
    _ensemble_reliability_objective,
    _temporal_validation_score,
)


def test_performance_metrics_include_return_risk_cost_and_benchmark() -> None:
    returns = np.asarray([0.01, -0.02, 0.03, 0.01])
    initial_cash = 1_000.0
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=4),
            "daily_return": returns,
            "benchmark_return": np.asarray([0.005, -0.01, 0.01, 0.0]),
            "nav": initial_cash * np.cumprod(1.0 + returns),
            "turnover": [0.5, 0.1, 0.0, 0.2],
            "trade_count": [2, 1, 0, 1],
            "broker_fees": [1.0, 0.5, 0.0, 0.5],
            "sell_tax": [0.0, 0.2, 0.0, 0.2],
            "available_cash": [100.0, 80.0, 70.0, 90.0],
            "pending_sale_cash": [0.0, 20.0, 0.0, 10.0],
        }
    )
    metrics = performance_metrics(frame, initial_cash=initial_cash)
    assert np.isclose(metrics["total_return"], np.prod(1.0 + returns) - 1.0)
    assert metrics["maximum_drawdown"] < 0
    assert metrics["total_turnover"] == 0.8
    assert metrics["trade_count"] == 4
    assert metrics["total_transaction_cost"] == 2.4
    assert metrics["cagr_elapsed_calendar_days"] == 3.0
    assert np.isclose(metrics["cagr_elapsed_years"], 3.0 / 365.25)
    assert metrics["annualization_periods_per_year"] == 252.0


def test_total_transaction_cost_includes_slippage() -> None:
    returns = np.asarray([0.01, -0.02, 0.03, 0.01])
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=4),
            "daily_return": returns,
            "benchmark_return": [0.005, -0.01, 0.01, 0.0],
            "nav": 1_000.0 * np.cumprod(1.0 + returns),
            "turnover": [0.5, 0.1, 0.0, 0.2],
            "trade_count": [2, 1, 0, 1],
            "broker_fees": [1.0, 0.5, 0.0, 0.5],
            "sell_tax": [0.0, 0.2, 0.0, 0.2],
            "available_cash": [100.0, 80.0, 70.0, 90.0],
            "pending_sale_cash": [0.0, 20.0, 0.0, 10.0],
        }
    )
    frame["slippage_cost"] = [0.4, 0.6, 0.0, 0.0]
    metrics = performance_metrics(frame, initial_cash=1_000.0)
    assert np.isclose(metrics["slippage_cost"], 1.0)
    assert np.isclose(metrics["total_transaction_cost"], 3.4)
    assert np.isclose(metrics["transaction_cost_to_initial"], 0.0034)
    assert "information_ratio" in metrics


def test_performance_metrics_reject_nav_return_mismatch() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=2),
            "daily_return": [0.01, 0.01],
            "benchmark_return": [0.0, 0.0],
            "nav": [1_010.0, 1_030.0],
            "turnover": [0.0, 0.0],
            "trade_count": [0, 0],
            "broker_fees": [0.0, 0.0],
            "sell_tax": [0.0, 0.0],
            "available_cash": [1_010.0, 1_030.0],
            "pending_sale_cash": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="NAV is inconsistent"):
        performance_metrics(frame, initial_cash=1_000.0)


def test_equal_weight_hold_days_do_not_create_hidden_rebalances() -> None:
    env = make_env(n_assets=2)
    daily, _ = evaluate_equal_weight(env, rebalance_every=20)
    assert daily.loc[0, "buy_notional"] > 0.0
    assert (daily.loc[1:, "turnover"] == 0.0).all()
    assert (daily.loc[1:, "trade_count"] == 0).all()


def test_portfolio_disagreement_uses_pairwise_total_variation() -> None:
    proposed = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    result = portfolio_disagreement(proposed)
    # Pairwise TV distances are 1.0, 0.5 and 0.5.
    assert np.isclose(result["disagreement_mean_pairwise_tv"], 2.0 / 3.0)
    assert result["disagreement_max_pairwise_tv"] == 1.0


def test_disagreement_gate_reduces_rebalancing_monotonically() -> None:
    values = [
        disagreement_rebalance_gate(
            disagreement,
            low_threshold=0.02,
            high_threshold=0.06,
            minimum_rebalance=0.10,
        )
        for disagreement in (0.01, 0.04, 0.08)
    ]
    assert np.allclose(values, [1.0, 0.55, 0.10])


def test_disagreement_controls_trade_not_seed_portfolio_weights() -> None:
    current = np.asarray([0.20, 0.30, 0.50])
    proposals = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.70, 0.20, 0.10],
            [0.20, 0.70, 0.10],
            [0.45, 0.45, 0.10],
        ]
    )
    policy = DisagreementPolicy(
        low_threshold=0.01,
        high_threshold=0.20,
        minimum_rebalance=0.10,
        low_quantile=0.25,
        high_quantile=0.90,
    )
    target, diagnostics = disagreement_controlled_target(
        proposals, current, policy
    )
    reordered, _ = disagreement_controlled_target(
        proposals[[4, 0, 3, 1, 2]], current, policy
    )
    consensus = proposals.mean(axis=0)
    assert np.allclose(target, reordered)
    assert diagnostics["rebalance_gate"] == 0.10
    assert np.allclose(target, current + 0.10 * (consensus - current))
    assert (
        diagnostics["adaptive_requested_change"]
        <= diagnostics["consensus_requested_change"]
    )


def test_reliability_weights_change_consensus_and_remain_permutation_safe() -> None:
    current = np.asarray([0.30, 0.30, 0.40])
    proposals = np.asarray(
        [[0.80, 0.10, 0.10], [0.10, 0.80, 0.10], [0.20, 0.20, 0.60]]
    )
    weights = np.asarray([0.70, 0.20, 0.10])
    target, diagnostics = disagreement_controlled_target(
        proposals, current, None, seed_weights=weights
    )
    reordered, _ = disagreement_controlled_target(
        proposals[[2, 0, 1]], current, None, seed_weights=weights[[2, 0, 1]]
    )
    assert np.allclose(target, reordered)
    assert np.allclose(target, np.average(proposals, axis=0, weights=weights))
    assert not np.allclose(target, proposals.mean(axis=0))
    assert diagnostics["effective_seed_count"] < 3.0


def test_regime_cap_moves_only_excess_stock_exposure_to_cash() -> None:
    weights = np.asarray([0.50, 0.30, 0.10, 0.10])
    capped = cap_stock_exposure(weights, 0.50)
    assert np.allclose(capped[:-1], [0.27777778, 0.16666667, 0.05555556])
    assert np.isclose(capped[-1], 0.50)
    assert np.isclose(capped.sum(), 1.0)


def test_regime_cap_does_not_leverage_a_defensive_portfolio() -> None:
    weights = np.asarray([0.20, 0.10, 0.70])
    assert np.allclose(cap_stock_exposure(weights, 0.50), weights)


def test_momentum_baseline_uses_last_completed_day_and_respects_top_k() -> None:
    env = make_env(n_assets=3)
    env.data.stock_features[1, :, 3] = [0.1, 0.3, 0.2]  # return_60d
    daily, _ = evaluate_cross_sectional_momentum(
        env,
        lookback_days=60,
        top_k=1,
        rebalance_every=20,
        cash_weight=0.05,
    )
    first = daily.iloc[0]
    assert first["target_B"] > 0.94
    assert first["target_A"] < 1e-6
    assert first["target_C"] < 1e-6
    assert np.isclose(first["target_CASH"], 0.05)


def test_inverse_volatility_baseline_is_long_only_and_preserves_cash() -> None:
    env = make_env(n_assets=3)
    daily, _ = evaluate_inverse_volatility(
        env, lookback_days=20, rebalance_every=20, cash_weight=0.05
    )
    first = daily.iloc[0]
    targets = first[["target_A", "target_B", "target_C", "target_CASH"]].to_numpy(
        dtype=float
    )
    assert np.all(targets >= 0.0)
    assert np.isclose(targets.sum(), 1.0)


def test_ensemble_reliability_stays_risk_adjusted() -> None:
    metrics = {
        "cagr": 0.30,
        "sharpe_ratio": 2.0,
        "annual_turnover": 4.0,
        "maximum_drawdown": -0.10,
    }
    config = PPOConfig(
        validation_objective="profit",
        validation_turnover_penalty=0.005,
        validation_drawdown_penalty=0.10,
    )
    assert np.isclose(_ensemble_reliability_objective(metrics, config), 1.97)


def test_temporal_validation_score_penalizes_unstable_years() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2021-01-04", "2021-01-05"]
            ),
            "daily_return": [0.10, 0.10, -0.05, -0.05],
        }
    )
    score, worst = _temporal_validation_score(daily)
    annual = np.asarray([0.21, -0.0975])
    assert np.isclose(score, annual.mean() - 0.5 * annual.std(ddof=0))
    assert np.isclose(worst, -0.0975)


def test_one_year_calibration_is_scored_by_quarter_not_degenerate_year() -> None:
    dates = pd.to_datetime(
        ["2023-01-03", "2023-04-03", "2023-07-03", "2023-10-02"]
    )
    daily = pd.DataFrame(
        {"date": dates, "daily_return": [0.10, -0.05, 0.02, 0.01]}
    )
    score, worst = _temporal_validation_score(daily)
    quarterly = np.asarray([0.10, -0.05, 0.02, 0.01])
    assert np.isclose(score, quarterly.mean() - 0.5 * quarterly.std(ddof=0))
    assert np.isclose(worst, -0.05)


def test_momentum_prior_strength_is_bounded() -> None:
    with pytest.raises(ValueError):
        PPOConfig(momentum_prior_strength=1.01).validate()


def test_paired_block_bootstrap_detects_constant_positive_alpha() -> None:
    baseline = np.zeros(60)
    strategy = np.full(60, 0.001)
    result = paired_circular_block_bootstrap(
        strategy, baseline, block_length=5, replications=500, seed=5
    )
    assert np.isclose(result["annualized_mean_active_return"], 0.252)
    assert result["ci_95_lower"] > 0
    assert result["two_sided_p_value"] < 0.01


def test_hac_test_detects_constant_positive_alpha() -> None:
    result = paired_hac_mean_test(
        np.full(60, 0.001), np.zeros(60), max_lag=5
    )
    assert np.isclose(result["annualized_mean_active_return"], 0.252)
    assert result["hac_ci_95_lower"] > 0
    assert result["hac_two_sided_p_value"] == 0.0


def test_hac_test_reports_return_noninferiority_separately() -> None:
    baseline = np.asarray([0.01, -0.01, 0.005, -0.005] * 30)
    strategy = baseline - 0.00001
    result = paired_hac_mean_test(
        strategy,
        baseline,
        max_lag=5,
        noninferiority_margin_annual=0.01,
    )
    assert result["annualized_mean_active_return"] < 0.0
    assert result["hac_two_sided_p_value"] == 0.0
    assert result["hac_noninferiority_p_value"] == 0.0


def test_disagreement_optimizer_penalizes_uncertain_assets_without_seed_weights() -> None:
    proposals = np.asarray(
        [
            [0.60, 0.30, 0.10],
            [0.20, 0.30, 0.50],
            [0.40, 0.30, 0.30],
        ]
    )
    current = np.asarray([0.40, 0.30, 0.30])
    policy = DisagreementOptimizationPolicy(
        disagreement_coefficient=1.0,
        turnover_regularization=0.0,
    )
    target, diagnostics = optimize_from_seed_disagreement(
        proposals, current, policy
    )
    reordered, _ = optimize_from_seed_disagreement(
        proposals[[2, 0, 1]], current, policy
    )
    assert np.allclose(target, reordered)
    assert np.isclose(target.sum(), 1.0)
    assert target[1] > proposals[:, 1].mean()
    assert (
        diagnostics["optimized_uncertainty_exposure"]
        < diagnostics["consensus_uncertainty_exposure"]
    )


def test_cross_seed_optimizer_turnover_penalty_stays_closer_to_current() -> None:
    proposals = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.25, 0.15],
            [0.65, 0.20, 0.15],
            [0.55, 0.30, 0.15],
            [0.75, 0.15, 0.10],
        ]
    )
    current = np.asarray([0.20, 0.40, 0.40])
    unregularized, _ = optimize_from_seed_disagreement(
        proposals,
        current,
        DisagreementOptimizationPolicy(0.0, 0.0),
    )
    regularized, _ = optimize_from_seed_disagreement(
        proposals,
        current,
        DisagreementOptimizationPolicy(0.0, 1.0),
    )
    assert 0.5 * np.abs(regularized - current).sum() < (
        0.5 * np.abs(unregularized - current).sum()
    )


def test_directional_optimizer_shrinks_trades_and_is_seed_symmetric() -> None:
    proposals = np.asarray(
        [
            [0.50, 0.30, 0.20],
            [0.48, 0.32, 0.20],
            [0.52, 0.28, 0.20],
            [0.46, 0.34, 0.20],
            [0.54, 0.26, 0.20],
        ]
    )
    current = np.asarray([0.40, 0.40, 0.20])
    policy = DirectionalDisagreementPolicy(
        low_threshold=0.01,
        high_threshold=0.20,
        minimum_rebalance=1.0,
        low_quantile=0.25,
        high_quantile=0.75,
        shrinkage_coefficient=2.0,
        turnover_regularization=0.5,
    )
    target, diagnostics = optimize_directional_disagreement(
        proposals, current, policy
    )
    reordered, _ = optimize_directional_disagreement(
        proposals[[4, 1, 3, 0, 2]], current, policy
    )
    consensus = proposals.mean(axis=0)
    assert np.allclose(target, reordered)
    assert np.isclose(target.sum(), 1.0)
    assert np.all(target >= 0.0)
    assert (
        0.5 * np.abs(target - current).sum()
        < 0.5 * np.abs(consensus - current).sum()
    )
    assert (
        diagnostics["directional_shrunk_trade_size"]
        < diagnostics["directional_mean_trade_size"]
    )


def test_all_consensus_operators_are_seed_permutation_invariant() -> None:
    proposals = np.asarray(
        [
            [0.55, 0.25, 0.15, 0.05],
            [0.50, 0.30, 0.15, 0.05],
            [0.45, 0.30, 0.20, 0.05],
            [0.60, 0.20, 0.15, 0.05],
            [0.40, 0.35, 0.20, 0.05],
        ]
    )
    for method in (
        "mean",
        "median",
        "trimmed_mean",
        "geometric",
        "geometric_sparsemax",
        "geometric_median",
    ):
        consensus = equal_seed_consensus(proposals, method)
        reordered = equal_seed_consensus(proposals[[4, 1, 3, 0, 2]], method)
        assert np.allclose(consensus, reordered)
        assert np.isclose(consensus.sum(), 1.0)
        assert np.all(consensus >= 0.0)


def test_geometric_median_resists_one_seed_outlier() -> None:
    proposals = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.69, 0.21, 0.10],
            [0.71, 0.19, 0.10],
            [0.68, 0.22, 0.10],
            [0.00, 0.10, 0.90],
        ]
    )
    mean = equal_seed_consensus(proposals, "mean")
    robust = equal_seed_consensus(proposals, "geometric_median")
    assert robust[0] > mean[0]
    assert robust[2] < mean[2]


def test_centered_l1_prox_and_turnover_projection_are_feasible() -> None:
    current = np.asarray([0.20, 0.30, 0.50])
    desired = np.asarray([0.70, 0.20, 0.10])
    penalized = centered_l1_simplex_prox(desired, current, 0.10)
    constrained = project_with_turnover_budget(desired, current, 0.15)
    assert np.isclose(penalized.sum(), 1.0)
    assert np.all(penalized >= 0.0)
    assert 0.5 * np.abs(penalized - current).sum() < (
        0.5 * np.abs(desired - current).sum()
    )
    assert 0.5 * np.abs(constrained - current).sum() <= 0.15 + 1e-9


def test_attached_soft_hard_asset_and_combined_strategies_are_seed_symmetric() -> None:
    proposals = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.65, 0.25, 0.10],
            [0.20, 0.70, 0.10],
            [0.68, 0.22, 0.10],
            [0.67, 0.23, 0.10],
        ]
    )
    current = np.asarray([0.30, 0.30, 0.40])
    policies = [
        AdaptiveRebalancePolicy("fixed_alpha", fixed_alpha=0.6),
        AdaptiveRebalancePolicy("soft", disagreement_k=4.0),
        AdaptiveRebalancePolicy(
            "hard_threshold",
            low_threshold=0.05,
            high_threshold=0.20,
            partial_rebalance=0.5,
        ),
        AdaptiveRebalancePolicy("asset_wise", disagreement_k=10.0),
        AdaptiveRebalancePolicy(
            "turnover_constraint",
            disagreement_k=4.0,
            base_turnover_budget=0.25,
        ),
        AdaptiveRebalancePolicy(
            "cost_penalty",
            disagreement_k=4.0,
            base_cost_penalty=0.05,
        ),
        AdaptiveRebalancePolicy(
            "combined",
            consensus_method="geometric_median",
            disagreement_k=4.0,
            base_turnover_budget=0.25,
            base_cost_penalty=0.05,
            asset_penalty=1.0,
        ),
    ]
    for policy in policies:
        target, diagnostics = adaptive_rebalance_target(
            proposals, current, policy
        )
        reordered, _ = adaptive_rebalance_target(
            proposals[[4, 2, 0, 3, 1]], current, policy
        )
        assert np.allclose(target, reordered)
        assert np.isclose(target.sum(), 1.0)
        assert np.all(target >= 0.0)
        assert diagnostics["effective_seed_count"] == 5.0
        assert diagnostics["adaptive_requested_change"] <= (
            diagnostics["consensus_requested_change"] + 1e-9
        )


def test_fixed_alpha_has_exact_target_turnover_scaling() -> None:
    proposals = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.65, 0.25, 0.10],
        ]
    )
    current = np.asarray([0.20, 0.30, 0.50])
    alpha = 0.4
    target, diagnostics = adaptive_rebalance_target(
        proposals,
        current,
        AdaptiveRebalancePolicy("fixed_alpha", fixed_alpha=alpha),
    )
    consensus = proposals.mean(axis=0)
    expected = current + alpha * (consensus - current)
    assert np.allclose(target, expected)
    assert np.isclose(
        diagnostics["adaptive_requested_change"],
        alpha * diagnostics["consensus_requested_change"],
    )


def test_geometric_consensus_requires_broad_seed_support() -> None:
    proposals = np.asarray(
        [
            [0.45, 0.45, 0.10],
            [0.45, 0.45, 0.10],
            [0.45, 0.45, 0.10],
            [0.45, 0.45, 0.10],
            [0.0001, 0.8999, 0.10],
        ]
    )
    arithmetic = equal_seed_consensus(proposals, "mean")
    geometric = equal_seed_consensus(proposals, "geometric")
    assert geometric[0] < arithmetic[0]
    assert geometric[1] > arithmetic[1]


def test_sparse_geometric_consensus_is_sparser_than_normalized_product() -> None:
    proposals = np.asarray(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.42, 0.28, 0.20, 0.10],
            [0.41, 0.29, 0.20, 0.10],
            [0.39, 0.31, 0.20, 0.10],
            [0.43, 0.27, 0.20, 0.10],
        ]
    )
    normalized = equal_seed_consensus(proposals, "geometric")
    sparse = equal_seed_consensus(proposals, "geometric_sparsemax")
    assert np.count_nonzero(sparse > 1e-10) < np.count_nonzero(normalized > 1e-10)
    assert sparse.max() > normalized.max()
