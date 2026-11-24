"""Reviewer-facing statistics computed only from frozen daily evaluations.

Nothing in this module trains or updates PPO parameters. The functions operate
on paired daily series and keep time-series uncertainty separate from the
cross-seed summaries produced by the caller.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def circular_block_indices(
    observations: int,
    block_length: int,
    replications: int,
    *,
    seed: int,
) -> np.ndarray:
    if observations < 2:
        raise ValueError("at least two observations are required")
    if block_length <= 0 or replications <= 0:
        raise ValueError("block_length and replications must be positive")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(observations / block_length))
    starts = rng.integers(0, observations, size=(replications, blocks))
    indices = (starts[:, :, None] + np.arange(block_length)) % observations
    return indices.reshape(replications, -1)[:, :observations]


def _metric_vector(
    returns: np.ndarray,
    *,
    years: float,
    periods_per_year: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    terminal_wealth = np.prod(1.0 + values, axis=1)
    total_return = terminal_wealth - 1.0
    cagr = np.power(terminal_wealth, 1.0 / years) - 1.0
    annual_mean = values.mean(axis=1) * periods_per_year
    annual_volatility = values.std(axis=1, ddof=1) * np.sqrt(periods_per_year)
    sharpe = np.divide(
        annual_mean,
        annual_volatility,
        out=np.full_like(annual_mean, np.nan),
        where=annual_volatility > 1e-15,
    )
    wealth_path = np.cumprod(1.0 + values, axis=1)
    wealth_path = np.concatenate(
        [np.ones((values.shape[0], 1), dtype=np.float64), wealth_path], axis=1
    )
    peaks = np.maximum.accumulate(wealth_path, axis=1)
    maximum_drawdown = np.min(wealth_path / peaks - 1.0, axis=1)
    return {
        "annualized_mean_return": annual_mean,
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": maximum_drawdown,
    }


def paired_metric_block_bootstrap(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    dates: Iterable[object],
    *,
    block_length: int = 20,
    replications: int = 20_000,
    seed: int = 20_260_828,
    periods_per_year: int = 252,
    batch_size: int = 1_000,
) -> pd.DataFrame:
    """Paired circular-block intervals for the main performance differences.

    The same resampled indices are applied to both strategies. CAGR uses the
    observed calendar duration; volatility and Sharpe use trading-session
    annualization. P-values are centered-bootstrap diagnostics and are treated
    as exploratory when the evaluation period has informed model development.
    """

    strategy = np.asarray(strategy_returns, dtype=np.float64)
    baseline = np.asarray(baseline_returns, dtype=np.float64)
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="raise"))
    if strategy.ndim != 1 or strategy.shape != baseline.shape:
        raise ValueError("paired returns must be one-dimensional and equally sized")
    if len(strategy) != len(date_index) or len(strategy) < 2:
        raise ValueError("dates must align with at least two paired returns")
    if not np.isfinite(strategy).all() or not np.isfinite(baseline).all():
        raise ValueError("paired returns must be finite")
    if not date_index.is_monotonic_increasing or date_index.has_duplicates:
        raise ValueError("dates must be unique and chronological")

    elapsed_days = max(int((date_index[-1] - date_index[0]).days), 1)
    years = elapsed_days / 365.25
    observed_strategy = _metric_vector(
        strategy, years=years, periods_per_year=periods_per_year
    )
    observed_baseline = _metric_vector(
        baseline, years=years, periods_per_year=periods_per_year
    )
    observed = {
        name: float(observed_strategy[name][0] - observed_baseline[name][0])
        for name in observed_strategy
    }

    indices = circular_block_indices(
        len(strategy), block_length, replications, seed=seed
    )
    boot = {name: np.empty(replications, dtype=np.float64) for name in observed}
    for start in range(0, replications, batch_size):
        stop = min(start + batch_size, replications)
        sample_indices = indices[start:stop]
        strategy_metrics = _metric_vector(
            strategy[sample_indices], years=years, periods_per_year=periods_per_year
        )
        baseline_metrics = _metric_vector(
            baseline[sample_indices], years=years, periods_per_year=periods_per_year
        )
        for name in boot:
            boot[name][start:stop] = strategy_metrics[name] - baseline_metrics[name]

    rows: list[dict[str, float | str]] = []
    for name, observed_difference in observed.items():
        finite = boot[name][np.isfinite(boot[name])]
        if finite.size == 0:
            lower = upper = p_value = float("nan")
        else:
            lower, upper = np.quantile(finite, [0.025, 0.975])
            centered = finite - finite.mean()
            p_value = (
                np.count_nonzero(np.abs(centered) >= abs(observed_difference)) + 1
            ) / (finite.size + 1)
        rows.append(
            {
                "metric": name,
                "observed_difference": observed_difference,
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "two_sided_p_value": float(p_value),
                "block_length": float(block_length),
                "replications": float(replications),
                "observations": float(len(strategy)),
                "elapsed_calendar_days": float(elapsed_days),
                "elapsed_years": float(years),
            }
        )
    return pd.DataFrame(rows)


def paired_additive_block_bootstrap(
    strategy_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    metric: str,
    scale: float = 1.0,
    block_length: int = 20,
    replications: int = 20_000,
    seed: int = 20_260_828,
) -> dict[str, float | str]:
    """Paired interval for an additive total such as turnover or transaction cost."""

    strategy = np.asarray(strategy_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    if strategy.ndim != 1 or strategy.shape != baseline.shape or strategy.size < 2:
        raise ValueError("paired values must be one-dimensional and equally sized")
    if not np.isfinite(strategy).all() or not np.isfinite(baseline).all():
        raise ValueError("paired values must be finite")
    difference = strategy - baseline
    observed = float(difference.sum() * scale)
    indices = circular_block_indices(
        len(difference), block_length, replications, seed=seed
    )
    bootstrap = difference[indices].sum(axis=1) * scale
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    centered = bootstrap - bootstrap.mean()
    p_value = (
        np.count_nonzero(np.abs(centered) >= abs(observed)) + 1
    ) / (replications + 1)
    return {
        "metric": metric,
        "observed_difference": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "two_sided_p_value": float(p_value),
        "block_length": float(block_length),
        "replications": float(replications),
        "observations": float(len(difference)),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values, preserving original order."""

    values = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return adjusted
    order = finite_positions[np.argsort(values[finite_positions])]
    running = 0.0
    count = len(order)
    for rank, position in enumerate(order):
        running = max(running, (count - rank) * values[position])
        adjusted[position] = min(running, 1.0)
    return adjusted


def spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    replications: int = 20_000,
    seed: int = 20_260_828,
) -> dict[str, float]:
    """Exploratory Spearman correlation and an iid pair-bootstrap interval."""

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 4:
        raise ValueError("at least four finite pairs are required")

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        rank_a = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
        rank_b = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
        if rank_a.std() <= 1e-15 or rank_b.std() <= 1e-15:
            return float("nan")
        return float(np.corrcoef(rank_a, rank_b)[0, 1])

    observed = correlation(left, right)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replications, dtype=np.float64)
    for index in range(replications):
        sample = rng.integers(0, len(left), len(left))
        bootstrap[index] = correlation(left[sample], right[sample])
    finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
    lower, upper = np.quantile(finite_bootstrap, [0.025, 0.975])
    return {
        "observations": float(len(left)),
        "spearman_rho": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "replications": float(replications),
    }
