from __future__ import annotations

import numpy as np
import pandas as pd

from vn30ppo.reviewer_statistics import (
    holm_adjust,
    paired_additive_block_bootstrap,
    paired_metric_block_bootstrap,
    spearman_with_bootstrap,
)


def test_paired_metric_bootstrap_reports_zero_for_identical_paths() -> None:
    returns = np.asarray([0.01, -0.005, 0.002, 0.004] * 15)
    dates = pd.bdate_range("2024-01-02", periods=len(returns))
    result = paired_metric_block_bootstrap(
        returns,
        returns,
        dates,
        block_length=5,
        replications=200,
        seed=5,
    )
    assert np.allclose(result["observed_difference"], 0.0)
    assert np.allclose(result["ci_95_lower"], 0.0)
    assert np.allclose(result["ci_95_upper"], 0.0)


def test_paired_additive_bootstrap_preserves_total_difference() -> None:
    result = paired_additive_block_bootstrap(
        np.full(30, 2.0),
        np.full(30, 1.0),
        metric="cost",
        block_length=5,
        replications=200,
        seed=10,
    )
    assert result["observed_difference"] == 30.0
    assert result["ci_95_lower"] == 30.0
    assert result["ci_95_upper"] == 30.0


def test_holm_adjust_is_monotonic_in_sorted_order() -> None:
    raw = np.asarray([0.03, 0.001, 0.02, np.nan])
    adjusted = holm_adjust(raw)
    assert np.isnan(adjusted[-1])
    assert adjusted[1] <= adjusted[2] <= adjusted[0]
    assert np.all(adjusted[:3] >= raw[:3])


def test_spearman_bootstrap_detects_monotonic_relation() -> None:
    x = np.arange(12, dtype=float)
    result = spearman_with_bootstrap(
        x,
        2.0 * x + 1.0,
        replications=200,
        seed=15,
    )
    assert np.isclose(result["spearman_rho"], 1.0)
    assert result["ci_95_lower"] > 0.9
