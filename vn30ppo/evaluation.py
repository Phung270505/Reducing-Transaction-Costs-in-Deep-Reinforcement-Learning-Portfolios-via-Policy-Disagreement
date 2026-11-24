from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import STOCK_FEATURE_NAMES
from .environment import VietnamStockEnv


class PredictableAgent(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        ...


ActionDiagnostics = dict[str, float | str]
ActionDecision = (
    np.ndarray
    | None
    | tuple[np.ndarray | None, ActionDiagnostics]
)
ActionFunction = Callable[[np.ndarray, VietnamStockEnv, int], ActionDecision]


@dataclass(slots=True, frozen=True)
class DisagreementPolicy:
    """Fixed validation-calibrated gate used during an untouched test run."""

    low_threshold: float
    high_threshold: float
    minimum_rebalance: float
    low_quantile: float
    high_quantile: float
    validation_score: float = float("nan")

    def validate(self) -> None:
        if self.low_threshold < 0 or self.high_threshold <= self.low_threshold:
            raise ValueError("disagreement thresholds must satisfy 0 <= low < high")
        if not 0 <= self.minimum_rebalance <= 1:
            raise ValueError("minimum_rebalance must be in [0, 1]")


@dataclass(slots=True, frozen=True)
class ReliabilityWeightedPolicy:
    """Calibration-only reliability weights plus a disagreement trade gate."""

    seed_weights: tuple[float, ...]
    seed_scores: tuple[float, ...]
    temperature: float
    gate_policy: DisagreementPolicy
    validation_score: float = float("nan")

    def validate(self) -> None:
        self.gate_policy.validate()
        weights = np.asarray(self.seed_weights, dtype=np.float64)
        scores = np.asarray(self.seed_scores, dtype=np.float64)
        if weights.ndim != 1 or len(weights) < 2 or scores.shape != weights.shape:
            raise ValueError("seed weights and scores must match at least two agents")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("seed weights must be finite and non-negative")
        if not np.isfinite(scores).all() or weights.sum() <= 0:
            raise ValueError("seed scores must be finite and weights must sum positive")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("seed weights must sum to one")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")


@dataclass(slots=True, frozen=True)
class DirectionalDisagreementPolicy:
    """Cost-aware optimizer built from equal-seed directional disagreement.

    Every seed has identical rights.  The policy regularizes the *change* from
    the current portfolio when seeds disagree about that change; it never
    converts seed performance or confidence into seed allocation weights.
    """

    low_threshold: float
    high_threshold: float
    minimum_rebalance: float
    low_quantile: float
    high_quantile: float
    shrinkage_coefficient: float = 0.0
    turnover_regularization: float = 0.0
    validation_score: float = float("nan")

    def validate(self) -> None:
        DisagreementPolicy(
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            minimum_rebalance=self.minimum_rebalance,
            low_quantile=self.low_quantile,
            high_quantile=self.high_quantile,
        ).validate()
        if self.shrinkage_coefficient < 0:
            raise ValueError("shrinkage_coefficient must be non-negative")
        if self.turnover_regularization < 0:
            raise ValueError("turnover_regularization must be non-negative")


@dataclass(slots=True, frozen=True)
class RegimeDisagreementPolicy:
    """Equal-seed disagreement gate with an independent VNINDEX risk cap.

    The disagreement gate controls whether the ensemble should rebalance.  A
    negative, fully observed VNINDEX trend separately caps stock exposure, so
    high disagreement cannot freeze a stale risky portfolio during a sell-off.
    No seed receives a reliability or performance weight.
    """

    low_threshold: float
    high_threshold: float
    minimum_rebalance: float
    low_quantile: float
    high_quantile: float
    trend_lookback_days: int = 60
    risk_off_stock_exposure: float = 0.5
    validation_score: float = float("nan")

    def validate(self) -> None:
        DisagreementPolicy(
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            minimum_rebalance=self.minimum_rebalance,
            low_quantile=self.low_quantile,
            high_quantile=self.high_quantile,
        ).validate()
        if self.trend_lookback_days < 20:
            raise ValueError("trend_lookback_days must be at least 20")
        if not 0 <= self.risk_off_stock_exposure <= 1:
            raise ValueError("risk_off_stock_exposure must be in [0, 1]")

    def disagreement_policy(self) -> DisagreementPolicy:
        return DisagreementPolicy(
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
            minimum_rebalance=self.minimum_rebalance,
            low_quantile=self.low_quantile,
            high_quantile=self.high_quantile,
            validation_score=self.validation_score,
        )


@dataclass(slots=True, frozen=True)
class DisagreementOptimizationPolicy:
    """Validation-selected robust optimizer with equal rights for every seed."""

    disagreement_coefficient: float
    turnover_regularization: float
    consensus_method: str = "mean"
    low_threshold: float | None = None
    high_threshold: float | None = None
    minimum_rebalance: float = 1.0
    validation_cagr: float = float("nan")

    def validate(self) -> None:
        if not np.isfinite(self.disagreement_coefficient):
            raise ValueError("disagreement_coefficient must be finite")
        if self.turnover_regularization < 0:
            raise ValueError("turnover_regularization must be non-negative")
        if self.consensus_method not in {
            "mean",
            "median",
            "trimmed_mean",
            "geometric",
            "geometric_sparsemax",
        }:
            raise ValueError("unknown equal-seed consensus method")
        thresholds = (self.low_threshold, self.high_threshold)
        if (thresholds[0] is None) != (thresholds[1] is None):
            raise ValueError("both disagreement thresholds must be set together")
        if thresholds[0] is not None and (
            thresholds[0] < 0 or thresholds[1] <= thresholds[0]
        ):
            raise ValueError("disagreement thresholds must satisfy 0 <= low < high")
        if not 0 <= self.minimum_rebalance <= 1:
            raise ValueError("minimum_rebalance must be in [0, 1]")


ATTACHED_REBALANCE_METHODS = {
    "fixed_alpha",
    "soft",
    "turnover_constraint",
    "cost_penalty",
    "asset_wise",
    "hard_threshold",
    "combined",
    "consensus",
}


@dataclass(slots=True, frozen=True)
class AdaptiveRebalancePolicy:
    """Equal-seed strategies requested in the dissertation design note."""

    method: str
    consensus_method: str = "mean"
    disagreement_k: float = 0.0
    base_turnover_budget: float = 1.0
    base_cost_penalty: float = 0.0
    asset_penalty: float = 0.0
    low_threshold: float | None = None
    high_threshold: float | None = None
    partial_rebalance: float = 0.5
    fixed_alpha: float = 1.0
    validation_score: float = float("nan")

    def validate(self) -> None:
        if self.method not in ATTACHED_REBALANCE_METHODS:
            raise ValueError(f"unknown adaptive rebalance method: {self.method}")
        if self.consensus_method not in {
            "mean",
            "median",
            "geometric_median",
        }:
            raise ValueError("unknown dissertation consensus method")
        for name in (
            "disagreement_k",
            "base_turnover_budget",
            "base_cost_penalty",
            "asset_penalty",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.base_turnover_budget > 1:
            raise ValueError("base_turnover_budget must be at most one")
        if not 0 <= self.partial_rebalance <= 1:
            raise ValueError("partial_rebalance must be in [0, 1]")
        if not 0 <= self.fixed_alpha <= 1:
            raise ValueError("fixed_alpha must be in [0, 1]")
        if self.method == "hard_threshold":
            if (
                self.low_threshold is None
                or self.high_threshold is None
                or self.low_threshold < 0
                or self.high_threshold <= self.low_threshold
            ):
                raise ValueError("hard thresholds must satisfy 0 <= low < high")


def portfolio_disagreement(proposed_weights: np.ndarray) -> dict[str, float]:
    """Measure dispersion between policies proposing portfolio distributions.

    Mean pairwise total variation is the primary, bounded [0, 1] signal.  The
    additional statistics are saved so the disagreement mechanism is auditable.
    """

    weights = np.asarray(proposed_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] < 2:
        raise ValueError("proposed_weights must contain at least two policies")
    if not np.isfinite(weights).all():
        raise ValueError("proposed portfolio weights must be finite")
    row_sums = weights.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("each proposed portfolio must have positive total weight")
    weights = weights / row_sums[:, None]
    consensus = weights.mean(axis=0)
    pairwise_tv = np.asarray(
        [
            0.5 * np.abs(weights[left] - weights[right]).sum()
            for left in range(len(weights))
            for right in range(left + 1, len(weights))
        ],
        dtype=np.float64,
    )
    asset_std = weights.std(axis=0, ddof=0)
    consensus_tv = 0.5 * np.abs(weights - consensus).sum(axis=1)
    return {
        "disagreement_mean_pairwise_tv": float(pairwise_tv.mean()),
        "disagreement_to_consensus_tv": float(consensus_tv.mean()),
        "disagreement_max_pairwise_tv": float(pairwise_tv.max()),
        "disagreement_rms": float(np.sqrt(np.mean((weights - consensus) ** 2))),
        "disagreement_mean_asset_std": float(asset_std.mean()),
        "disagreement_max_asset_std": float(asset_std.max()),
        "disagreement_cash_std": float(asset_std[-1]),
    }


def disagreement_rebalance_gate(
    disagreement: float,
    *,
    low_threshold: float,
    high_threshold: float,
    minimum_rebalance: float,
) -> float:
    """Map low disagreement to full rebalancing and high to a minimum gate."""

    if high_threshold <= low_threshold:
        raise ValueError("high_threshold must be greater than low_threshold")
    if not 0 <= minimum_rebalance <= 1:
        raise ValueError("minimum_rebalance must be in [0, 1]")
    confidence = np.clip(
        (high_threshold - disagreement) / (high_threshold - low_threshold),
        0.0,
        1.0,
    )
    return float(minimum_rebalance + (1.0 - minimum_rebalance) * confidence)


def disagreement_controlled_target(
    proposed_weights: np.ndarray,
    current_weights: np.ndarray,
    policy: DisagreementPolicy | None,
    *,
    initial_deployment: bool = False,
    seed_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Create one equal-seed consensus trade controlled by disagreement.

    With no ``seed_weights`` this is the arithmetic mean baseline. The proposed
    strategy supplies calibration-only reliability weights; disagreement then
    controls how far the portfolio may move from observable holdings toward
    that regularized consensus.
    """

    proposals = np.asarray(proposed_weights, dtype=np.float64)
    current = np.asarray(current_weights, dtype=np.float64)
    if proposals.ndim != 2 or proposals.shape[0] < 2:
        raise ValueError("proposed_weights must contain at least two policies")
    if current.shape != (proposals.shape[1],):
        raise ValueError("current_weights and proposals have incompatible shapes")
    if not np.isfinite(proposals).all() or not np.isfinite(current).all():
        raise ValueError("portfolio weights must be finite")
    if (proposals < 0).any() or (current < 0).any():
        raise ValueError("portfolio weights must be non-negative")
    proposal_sums = proposals.sum(axis=1)
    if (proposal_sums <= 0).any() or current.sum() <= 0:
        raise ValueError("every portfolio must have a positive sum")
    proposals = proposals / proposal_sums[:, None]
    current = current / current.sum()

    if seed_weights is None:
        normalized_seed_weights = np.full(
            proposals.shape[0], 1.0 / proposals.shape[0]
        )
    else:
        normalized_seed_weights = np.asarray(seed_weights, dtype=np.float64)
        if normalized_seed_weights.shape != (proposals.shape[0],):
            raise ValueError("seed_weights must match the number of proposals")
        if (
            not np.isfinite(normalized_seed_weights).all()
            or (normalized_seed_weights < 0).any()
            or normalized_seed_weights.sum() <= 0
        ):
            raise ValueError("seed_weights must be finite and non-negative")
        normalized_seed_weights /= normalized_seed_weights.sum()
    consensus = np.average(
        proposals, axis=0, weights=normalized_seed_weights
    )
    diagnostics = portfolio_disagreement(proposals)
    if policy is None or initial_deployment:
        gate = 1.0
    else:
        policy.validate()
        gate = disagreement_rebalance_gate(
            diagnostics["disagreement_mean_pairwise_tv"],
            low_threshold=policy.low_threshold,
            high_threshold=policy.high_threshold,
            minimum_rebalance=policy.minimum_rebalance,
        )
    target = current + gate * (consensus - current)
    target = np.clip(target, 0.0, None)
    target /= target.sum()
    consensus_change = float(0.5 * np.abs(consensus - current).sum())
    adaptive_change = float(0.5 * np.abs(target - current).sum())
    if adaptive_change > consensus_change + 1e-10:
        raise RuntimeError("disagreement gate increased the requested turnover")
    diagnostics.update(
        {
            "rebalance_gate": float(gate),
            "consensus_requested_change": consensus_change,
            "adaptive_requested_change": adaptive_change,
            "disagreement_turnover_reduction": consensus_change - adaptive_change,
            "maximum_seed_weight": float(normalized_seed_weights.max()),
            "effective_seed_count": float(
                1.0 / np.sum(normalized_seed_weights**2)
            ),
        }
    )
    if policy is not None:
        diagnostics.update(
            {
                "disagreement_low_threshold": policy.low_threshold,
                "disagreement_high_threshold": policy.high_threshold,
            }
        )
    return target, diagnostics


def cap_stock_exposure(weights: np.ndarray, maximum_stock_exposure: float) -> np.ndarray:
    """Move stock weight above a fixed cap to cash without changing stock mix."""

    portfolio = np.asarray(weights, dtype=np.float64).copy()
    if portfolio.ndim != 1 or portfolio.size < 2 or not np.isfinite(portfolio).all():
        raise ValueError("weights must be a finite stock-plus-cash vector")
    if not 0 <= maximum_stock_exposure <= 1:
        raise ValueError("maximum_stock_exposure must be in [0, 1]")
    portfolio = np.clip(portfolio, 0.0, None)
    total = float(portfolio.sum())
    if total <= 1e-12:
        raise ValueError("portfolio weights must have a positive sum")
    portfolio /= total
    stock_exposure = float(portfolio[:-1].sum())
    if stock_exposure > maximum_stock_exposure + 1e-12:
        portfolio[:-1] *= maximum_stock_exposure / stock_exposure
        portfolio[-1] = 1.0 - maximum_stock_exposure
    return portfolio / portfolio.sum()


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto non-negative portfolio weights summing to one."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("values must be a finite non-empty vector")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    positive = ordered - cumulative / np.arange(1, vector.size + 1) > 0
    if not positive.any():
        return np.full(vector.size, 1.0 / vector.size)
    rho = np.flatnonzero(positive)[-1]
    threshold = cumulative[rho] / (rho + 1.0)
    projected = np.maximum(vector - threshold, 0.0)
    return projected / projected.sum()


def optimize_directional_disagreement(
    proposed_weights: np.ndarray,
    current_weights: np.ndarray,
    policy: DirectionalDisagreementPolicy,
    *,
    initial_deployment: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve a symmetric, cost-aware trade from the five PPO proposals.

    Let ``delta_i = w_i - w_current`` be seed ``i``'s proposed trade.  For
    every asset, the equal-seed mean trade is soft-thresholded by the standard
    deviation of those trades and then shrunk toward zero by a turnover
    penalty.  In closed form before the simplex projection this is

        delta* = soft_threshold(mean(delta), kappa * std(delta)) / (1 + eta)

    High directional disagreement therefore creates an asset-specific no-trade
    region instead of changing the percentage assigned to any seed.  The
    validation-calibrated global TV gate is applied after this optimization.
    """

    policy.validate()
    proposals = np.asarray(proposed_weights, dtype=np.float64)
    current = np.asarray(current_weights, dtype=np.float64)
    if proposals.ndim != 2 or proposals.shape[0] < 2:
        raise ValueError("proposed_weights must contain at least two seed policies")
    if current.shape != (proposals.shape[1],):
        raise ValueError("current_weights must match the proposal action dimension")
    if not np.isfinite(proposals).all() or not np.isfinite(current).all():
        raise ValueError("portfolio weights must be finite")
    proposal_totals = proposals.sum(axis=1)
    if np.any(proposal_totals <= 0) or current.sum() <= 0:
        raise ValueError("portfolio weights must have positive sums")

    proposals = proposals / proposal_totals[:, None]
    current = current / current.sum()
    proposed_trades = proposals - current[None, :]
    mean_trade = proposed_trades.mean(axis=0)
    directional_uncertainty = proposed_trades.std(axis=0, ddof=0)
    threshold = policy.shrinkage_coefficient * directional_uncertainty
    shrunk_trade = np.sign(mean_trade) * np.maximum(
        np.abs(mean_trade) - threshold,
        0.0,
    )
    # Initial capital should not remain in cash merely because eta encodes a
    # preference for the previous portfolio.  Kappa still governs uncertainty.
    eta = 0.0 if initial_deployment else policy.turnover_regularization
    regularized_trade = shrunk_trade / (1.0 + eta)
    optimized_target = project_to_simplex(current + regularized_trade)

    disagreement = portfolio_disagreement(proposals)
    gate = (
        1.0
        if initial_deployment
        else disagreement_rebalance_gate(
            disagreement["disagreement_mean_pairwise_tv"],
            low_threshold=policy.low_threshold,
            high_threshold=policy.high_threshold,
            minimum_rebalance=policy.minimum_rebalance,
        )
    )
    target = project_to_simplex(
        current + gate * (optimized_target - current)
    )
    consensus = proposals.mean(axis=0)
    consensus_change = float(0.5 * np.abs(consensus - current).sum())
    optimized_change = float(0.5 * np.abs(target - current).sum())
    diagnostics = {
        **disagreement,
        "rebalance_gate": float(gate),
        "directional_shrinkage_coefficient": policy.shrinkage_coefficient,
        "directional_turnover_regularization": policy.turnover_regularization,
        "disagreement_low_threshold": policy.low_threshold,
        "disagreement_high_threshold": policy.high_threshold,
        "consensus_requested_change": consensus_change,
        "adaptive_requested_change": optimized_change,
        "disagreement_turnover_reduction": consensus_change - optimized_change,
        "directional_mean_trade_size": float(0.5 * np.abs(mean_trade).sum()),
        "directional_shrunk_trade_size": float(
            0.5 * np.abs(shrunk_trade).sum()
        ),
        "directional_optimized_trade_size": float(
            0.5 * np.abs(target - current).sum()
        ),
        "directional_suppressed_assets": float(
            np.count_nonzero(
                (np.abs(mean_trade) > 1e-12)
                & (np.abs(shrunk_trade) <= 1e-12)
            )
        ),
        "directional_uncertainty_exposure": float(
            np.dot(target, directional_uncertainty)
        ),
    }
    return target, diagnostics


def equal_seed_consensus(
    proposed_weights: np.ndarray, method: str
) -> np.ndarray:
    """Aggregate seed proposals symmetrically, without seed-level weights."""

    proposals = np.asarray(proposed_weights, dtype=np.float64)
    if proposals.ndim != 2 or proposals.shape[0] < 2:
        raise ValueError("proposed_weights must contain at least two policies")
    proposals = proposals / proposals.sum(axis=1, keepdims=True)
    if method == "mean":
        consensus = proposals.mean(axis=0)
    elif method == "median":
        consensus = np.median(proposals, axis=0)
    elif method == "geometric_median":
        consensus = proposals.mean(axis=0)
        for _ in range(200):
            distances = np.linalg.norm(proposals - consensus, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 1e-12:
                consensus = proposals[nearest]
                break
            inverse_distances = 1.0 / distances
            updated = np.average(
                proposals, axis=0, weights=inverse_distances
            )
            if np.linalg.norm(updated - consensus) <= 1e-12:
                consensus = updated
                break
            consensus = updated
    elif method == "trimmed_mean":
        if proposals.shape[0] < 5:
            raise ValueError("trimmed_mean requires at least five seed policies")
        consensus = np.sort(proposals, axis=0)[1:-1].mean(axis=0)
    elif method in {"geometric", "geometric_sparsemax"}:
        # Product-of-experts consensus: one seed assigning nearly zero weight
        # strongly reduces the asset, emphasizing unanimous selections.
        log_consensus = np.log(np.clip(proposals, 1e-8, None)).mean(axis=0)
        log_consensus -= log_consensus.max()
        consensus = np.exp(log_consensus)
    else:
        raise ValueError(f"Unknown equal-seed consensus method: {method}")
    if method == "geometric_sparsemax":
        # Sparsemax-style Euclidean projection of relative product-of-experts
        # support. It keeps seed symmetry while requiring broad support for an
        # asset to survive in the investable consensus.
        return project_to_simplex(consensus)
    consensus = np.clip(consensus, 0.0, None)
    total = float(consensus.sum())
    if total <= 1e-12:
        return np.full(proposals.shape[1], 1.0 / proposals.shape[1])
    return consensus / total


def centered_l1_simplex_prox(
    values: np.ndarray,
    current_weights: np.ndarray,
    penalties: float | np.ndarray,
) -> np.ndarray:
    """Exact prox for weighted L1 trading cost on the probability simplex.

    Solves ``min 0.5||w-values||² + sum_j penalty_j|w_j-current_j|``
    subject to ``w>=0`` and ``sum(w)=1`` using the monotone simplex dual.
    """

    target = np.asarray(values, dtype=np.float64)
    current = np.asarray(current_weights, dtype=np.float64)
    penalty = np.broadcast_to(
        np.asarray(penalties, dtype=np.float64), target.shape
    ).copy()
    if target.ndim != 1 or current.shape != target.shape:
        raise ValueError("values and current_weights must be matching vectors")
    if (
        not np.isfinite(target).all()
        or not np.isfinite(current).all()
        or not np.isfinite(penalty).all()
        or (penalty < 0).any()
        or current.sum() <= 0
    ):
        raise ValueError("prox inputs must be finite with non-negative penalties")
    current = np.clip(current, 0.0, None)
    current /= current.sum()

    def solution(dual: float) -> np.ndarray:
        shifted = target - dual
        difference = shifted - current
        prox = np.where(
            difference > penalty,
            shifted - penalty,
            np.where(
                difference < -penalty,
                shifted + penalty,
                current,
            ),
        )
        return np.maximum(prox, 0.0)

    lower, upper = -1.0, 1.0
    while solution(lower).sum() < 1.0:
        lower *= 2.0
    while solution(upper).sum() > 1.0:
        upper *= 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if solution(midpoint).sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    result = solution(0.5 * (lower + upper))
    return result / result.sum()


def project_with_turnover_budget(
    desired_weights: np.ndarray,
    current_weights: np.ndarray,
    turnover_budget: float,
) -> np.ndarray:
    """Euclidean projection onto simplex portfolios within an L1 turnover cap."""

    desired = project_to_simplex(np.asarray(desired_weights, dtype=np.float64))
    current = project_to_simplex(np.asarray(current_weights, dtype=np.float64))
    if not np.isfinite(turnover_budget) or not 0 <= turnover_budget <= 1:
        raise ValueError("turnover_budget must be finite and in [0, 1]")
    desired_turnover = float(0.5 * np.abs(desired - current).sum())
    if desired_turnover <= turnover_budget + 1e-12:
        return desired
    if turnover_budget <= 1e-12:
        return current
    lower, upper = 0.0, 1.0
    while True:
        candidate = centered_l1_simplex_prox(desired, current, upper)
        if 0.5 * np.abs(candidate - current).sum() <= turnover_budget:
            break
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        candidate = centered_l1_simplex_prox(desired, current, midpoint)
        if 0.5 * np.abs(candidate - current).sum() > turnover_budget:
            lower = midpoint
        else:
            upper = midpoint
    return centered_l1_simplex_prox(desired, current, upper)


def adaptive_rebalance_target(
    proposed_weights: np.ndarray,
    current_weights: np.ndarray,
    policy: AdaptiveRebalancePolicy,
    *,
    initial_deployment: bool = False,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Apply one of the disagreement strategies supplied by the user."""

    policy.validate()
    proposals = np.asarray(proposed_weights, dtype=np.float64)
    current = np.asarray(current_weights, dtype=np.float64)
    if proposals.ndim != 2 or proposals.shape[0] < 2:
        raise ValueError("proposed_weights must contain at least two seed policies")
    if current.shape != (proposals.shape[1],):
        raise ValueError("current_weights must match proposal dimension")
    if not np.isfinite(proposals).all() or not np.isfinite(current).all():
        raise ValueError("portfolio weights must be finite")
    proposals = proposals / proposals.sum(axis=1, keepdims=True)
    current = project_to_simplex(current)
    consensus = equal_seed_consensus(proposals, policy.consensus_method)
    diagnostics = portfolio_disagreement(proposals)
    disagreement = diagnostics["disagreement_to_consensus_tv"]
    asset_uncertainty = proposals.std(axis=0, ddof=0)
    consensus_turnover = float(0.5 * np.abs(consensus - current).sum())
    alpha = 1.0
    turnover_budget = 1.0
    effective_penalty = 0.0

    if initial_deployment or policy.method == "consensus":
        target = consensus
    elif policy.method == "fixed_alpha":
        alpha = float(policy.fixed_alpha)
        target = current + alpha * (consensus - current)
    elif policy.method == "soft":
        alpha = float(np.exp(-policy.disagreement_k * disagreement))
        target = current + alpha * (consensus - current)
    elif policy.method == "hard_threshold":
        if disagreement < float(policy.low_threshold):
            alpha = 1.0
        elif disagreement < float(policy.high_threshold):
            alpha = policy.partial_rebalance
        else:
            alpha = 0.0
        target = current + alpha * (consensus - current)
    elif policy.method == "turnover_constraint":
        turnover_budget = float(
            policy.base_turnover_budget
            * np.exp(-policy.disagreement_k * disagreement)
        )
        target = project_with_turnover_budget(
            consensus, current, turnover_budget
        )
        alpha = (
            min(1.0, turnover_budget / consensus_turnover)
            if consensus_turnover > 1e-12
            else 1.0
        )
    elif policy.method == "cost_penalty":
        effective_penalty = float(
            policy.base_cost_penalty
            * (1.0 + policy.disagreement_k * disagreement)
        )
        target = centered_l1_simplex_prox(
            consensus, current, effective_penalty
        )
    elif policy.method == "asset_wise":
        asset_alpha = np.exp(-policy.disagreement_k * asset_uncertainty)
        target = project_to_simplex(
            current + asset_alpha * (consensus - current)
        )
        alpha = float(asset_alpha.mean())
    elif policy.method == "combined":
        effective_penalty = float(
            policy.base_cost_penalty * disagreement
        )
        penalties = effective_penalty + policy.asset_penalty * asset_uncertainty
        penalized = centered_l1_simplex_prox(
            consensus, current, penalties
        )
        turnover_budget = float(
            policy.base_turnover_budget
            * np.exp(-policy.disagreement_k * disagreement)
        )
        target = project_with_turnover_budget(
            penalized, current, turnover_budget
        )
    else:  # pragma: no cover - validate() makes this unreachable.
        raise ValueError(f"unsupported adaptive method: {policy.method}")

    target = project_to_simplex(target)
    adaptive_turnover = float(0.5 * np.abs(target - current).sum())
    diagnostics.update(
        {
            "adaptive_method": policy.method,
            "adaptive_consensus_method": policy.consensus_method,
            "rebalance_gate": alpha,
            "adaptive_disagreement_k": policy.disagreement_k,
            "adaptive_turnover_budget": turnover_budget,
            "adaptive_effective_cost_penalty": effective_penalty,
            "adaptive_mean_asset_uncertainty": float(asset_uncertainty.mean()),
            "adaptive_max_asset_uncertainty": float(asset_uncertainty.max()),
            "consensus_requested_change": consensus_turnover,
            "adaptive_requested_change": adaptive_turnover,
            "disagreement_turnover_reduction": (
                consensus_turnover - adaptive_turnover
            ),
            "maximum_seed_weight": 1.0 / proposals.shape[0],
            "effective_seed_count": float(proposals.shape[0]),
        }
    )
    return target, diagnostics


def optimize_from_seed_disagreement(
    proposed_weights: np.ndarray,
    current_weights: np.ndarray,
    policy: DisagreementOptimizationPolicy,
) -> tuple[np.ndarray, dict[str, float]]:
    """Find a robust portfolio directly from cross-seed disagreement.

    All seeds enter symmetrically.  The solution minimizes

        0.5||w - mean(W)||² + lambda * std(W)'w
        + 0.5 * eta||w - w_current||²,  w in the probability simplex.

    Thus no validation score is converted into a percentage for any seed.
    """

    policy.validate()
    proposals = np.asarray(proposed_weights, dtype=np.float64)
    current = np.asarray(current_weights, dtype=np.float64)
    if proposals.ndim != 2 or proposals.shape[0] < 2:
        raise ValueError("proposed_weights must contain at least two seed policies")
    if current.shape != (proposals.shape[1],):
        raise ValueError("current_weights must match the proposal action dimension")
    if not np.isfinite(proposals).all() or not np.isfinite(current).all():
        raise ValueError("portfolio weights must be finite")
    proposals = proposals / proposals.sum(axis=1, keepdims=True)
    current = current / max(current.sum(), 1e-12)
    consensus = equal_seed_consensus(proposals, policy.consensus_method)
    asset_uncertainty = proposals.std(axis=0, ddof=0)
    eta = policy.turnover_regularization
    unconstrained = (
        consensus
        + eta * current
        - policy.disagreement_coefficient * asset_uncertainty
    ) / (1.0 + eta)
    robust_target = project_to_simplex(unconstrained)
    disagreement_diagnostics = portfolio_disagreement(proposals)
    if policy.low_threshold is None:
        gate = 1.0
    else:
        gate = disagreement_rebalance_gate(
            disagreement_diagnostics["disagreement_mean_pairwise_tv"],
            low_threshold=policy.low_threshold,
            high_threshold=policy.high_threshold,
            minimum_rebalance=policy.minimum_rebalance,
        )
    target = current + gate * (robust_target - current)
    target = project_to_simplex(target)
    diagnostics = {
        **disagreement_diagnostics,
        "optimizer_disagreement_coefficient": policy.disagreement_coefficient,
        "optimizer_turnover_regularization": eta,
        "optimizer_consensus_method": policy.consensus_method,
        "disagreement_low_threshold": (
            float(policy.low_threshold)
            if policy.low_threshold is not None
            else float("nan")
        ),
        "disagreement_high_threshold": (
            float(policy.high_threshold)
            if policy.high_threshold is not None
            else float("nan")
        ),
        "rebalance_gate": gate,
        "consensus_uncertainty_exposure": float(
            np.dot(consensus, asset_uncertainty)
        ),
        "robust_uncertainty_exposure": float(
            np.dot(robust_target, asset_uncertainty)
        ),
        "optimized_uncertainty_exposure": float(
            np.dot(target, asset_uncertainty)
        ),
        "consensus_requested_change": float(
            0.5 * np.abs(consensus - current).sum()
        ),
        "optimized_requested_change": float(
            0.5 * np.abs(target - current).sum()
        ),
        "optimizer_active_assets": float(np.count_nonzero(target > 1e-8)),
    }
    return target, diagnostics


def _adaptive_rebalance_action(
    observation: np.ndarray,
    env: VietnamStockEnv,
    step: int,
    agents: Sequence[PredictableAgent],
    policy: AdaptiveRebalancePolicy,
) -> ActionDecision:
    intervals = {
        int(getattr(getattr(agent, "config", None), "rebalance_interval", 1))
        for agent in agents
    }
    rebalance_interval = intervals.pop() if len(intervals) == 1 else 1
    if step % max(rebalance_interval, 1):
        return None, {
            "disagreement_mean_pairwise_tv": float("nan"),
            "disagreement_to_consensus_tv": float("nan"),
            "disagreement_max_pairwise_tv": float("nan"),
            "disagreement_rms": float("nan"),
            "disagreement_mean_asset_std": float("nan"),
            "disagreement_max_asset_std": float("nan"),
            "disagreement_cash_std": float("nan"),
            "adaptive_method": policy.method,
            "adaptive_consensus_method": policy.consensus_method,
            "rebalance_gate": 0.0,
            "adaptive_disagreement_k": policy.disagreement_k,
            "adaptive_turnover_budget": 0.0,
            "adaptive_effective_cost_penalty": 0.0,
            "adaptive_mean_asset_uncertainty": float("nan"),
            "adaptive_max_asset_uncertainty": float("nan"),
            "consensus_requested_change": 0.0,
            "adaptive_requested_change": 0.0,
            "disagreement_turnover_reduction": 0.0,
            "maximum_seed_weight": 1.0 / len(agents),
            "effective_seed_count": float(len(agents)),
        }
    proposals = np.stack(
        [env.action_to_weights(agent.predict(observation)) for agent in agents]
    )
    current = env.current_hold_weights()
    target, diagnostics = adaptive_rebalance_target(
        proposals,
        current,
        policy,
        initial_deployment=step == 0,
    )
    return env.weights_to_action(target), diagnostics


def _directional_disagreement_action(
    observation: np.ndarray,
    env: VietnamStockEnv,
    step: int,
    agents: Sequence[PredictableAgent],
    policy: DirectionalDisagreementPolicy,
) -> ActionDecision:
    intervals = {
        int(getattr(getattr(agent, "config", None), "rebalance_interval", 1))
        for agent in agents
    }
    rebalance_interval = intervals.pop() if len(intervals) == 1 else 1
    if step % max(rebalance_interval, 1):
        return None, {
            "disagreement_mean_pairwise_tv": float("nan"),
            "disagreement_max_pairwise_tv": float("nan"),
            "disagreement_rms": float("nan"),
            "disagreement_mean_asset_std": float("nan"),
            "disagreement_max_asset_std": float("nan"),
            "disagreement_cash_std": float("nan"),
            "rebalance_gate": 0.0,
            "directional_shrinkage_coefficient": policy.shrinkage_coefficient,
            "directional_turnover_regularization": (
                policy.turnover_regularization
            ),
            "disagreement_low_threshold": policy.low_threshold,
            "disagreement_high_threshold": policy.high_threshold,
            "consensus_requested_change": 0.0,
            "adaptive_requested_change": 0.0,
            "disagreement_turnover_reduction": 0.0,
            "directional_mean_trade_size": 0.0,
            "directional_shrunk_trade_size": 0.0,
            "directional_optimized_trade_size": 0.0,
            "directional_suppressed_assets": float("nan"),
            "directional_uncertainty_exposure": float("nan"),
        }
    proposals = np.stack(
        [env.action_to_weights(agent.predict(observation)) for agent in agents]
    )
    current = env.current_hold_weights()
    target, diagnostics = optimize_directional_disagreement(
        proposals,
        current,
        policy,
        initial_deployment=step == 0,
    )
    return env.weights_to_action(target), diagnostics


def _optimized_disagreement_action(
    observation: np.ndarray,
    env: VietnamStockEnv,
    step: int,
    agents: Sequence[PredictableAgent],
    policy: DisagreementOptimizationPolicy,
) -> ActionDecision:
    intervals = {
        int(getattr(getattr(agent, "config", None), "rebalance_interval", 1))
        for agent in agents
    }
    rebalance_interval = intervals.pop() if len(intervals) == 1 else 1
    if step % max(rebalance_interval, 1):
        return None, {
            "disagreement_mean_pairwise_tv": float("nan"),
            "disagreement_max_pairwise_tv": float("nan"),
            "disagreement_rms": float("nan"),
            "disagreement_mean_asset_std": float("nan"),
            "disagreement_max_asset_std": float("nan"),
            "disagreement_cash_std": float("nan"),
            "optimizer_disagreement_coefficient": policy.disagreement_coefficient,
            "optimizer_turnover_regularization": policy.turnover_regularization,
            "optimizer_consensus_method": policy.consensus_method,
            "disagreement_low_threshold": (
                float(policy.low_threshold)
                if policy.low_threshold is not None
                else float("nan")
            ),
            "disagreement_high_threshold": (
                float(policy.high_threshold)
                if policy.high_threshold is not None
                else float("nan")
            ),
            "rebalance_gate": 0.0,
            "consensus_uncertainty_exposure": float("nan"),
            "robust_uncertainty_exposure": float("nan"),
            "optimized_uncertainty_exposure": float("nan"),
            "consensus_requested_change": 0.0,
            "optimized_requested_change": 0.0,
            "optimizer_active_assets": float("nan"),
        }
    proposals = np.stack(
        [env.action_to_weights(agent.predict(observation)) for agent in agents]
    )
    current = env.current_hold_weights()
    effective_policy = policy
    if step == 0 and policy.turnover_regularization > 0:
        # Deploy initial capital without treating the starting cash position as
        # a preferred portfolio.  Disagreement aversion remains active.
        effective_policy = DisagreementOptimizationPolicy(
            disagreement_coefficient=policy.disagreement_coefficient,
            turnover_regularization=0.0,
            consensus_method=policy.consensus_method,
            low_threshold=None,
            high_threshold=None,
            minimum_rebalance=1.0,
            validation_cagr=policy.validation_cagr,
        )
    target, diagnostics = optimize_from_seed_disagreement(
        proposals, current, effective_policy
    )
    diagnostics["optimizer_turnover_regularization"] = (
        policy.turnover_regularization
    )
    if step == 0:
        diagnostics["disagreement_low_threshold"] = (
            float(policy.low_threshold)
            if policy.low_threshold is not None
            else float("nan")
        )
        diagnostics["disagreement_high_threshold"] = (
            float(policy.high_threshold)
            if policy.high_threshold is not None
            else float("nan")
        )
        diagnostics["rebalance_gate"] = 1.0
    return env.weights_to_action(target), diagnostics


def _ensemble_action(
    observation: np.ndarray,
    env: VietnamStockEnv,
    step: int,
    agents: Sequence[PredictableAgent],
    policy: DisagreementPolicy | None,
    seed_weights: np.ndarray | None = None,
) -> ActionDecision:
    intervals = {
        int(getattr(getattr(agent, "config", None), "rebalance_interval", 1))
        for agent in agents
    }
    rebalance_interval = intervals.pop() if len(intervals) == 1 else 1
    if step % max(rebalance_interval, 1):
        diagnostics = {
            "disagreement_mean_pairwise_tv": float("nan"),
            "disagreement_max_pairwise_tv": float("nan"),
            "disagreement_rms": float("nan"),
            "disagreement_mean_asset_std": float("nan"),
            "disagreement_max_asset_std": float("nan"),
            "disagreement_cash_std": float("nan"),
            "rebalance_gate": 0.0,
            "consensus_requested_change": 0.0,
            "adaptive_requested_change": 0.0,
            "disagreement_turnover_reduction": 0.0,
            "maximum_seed_weight": (
                float(np.max(seed_weights))
                if seed_weights is not None
                else 1.0 / len(agents)
            ),
            "effective_seed_count": (
                float(1.0 / np.sum(np.asarray(seed_weights) ** 2))
                if seed_weights is not None
                else float(len(agents))
            ),
        }
        if policy is not None:
            diagnostics.update(
                {
                    "disagreement_low_threshold": policy.low_threshold,
                    "disagreement_high_threshold": policy.high_threshold,
                }
            )
        return None, diagnostics
    proposals = np.stack(
        [env.action_to_weights(agent.predict(observation)) for agent in agents]
    )
    current = env.current_hold_weights()
    target, diagnostics = disagreement_controlled_target(
        proposals,
        current,
        policy,
        # Disagreement controls rebalancing, not initial deployment of cash.
        initial_deployment=step == 0,
        seed_weights=seed_weights,
    )
    return env.weights_to_action(target), diagnostics


def _regime_disagreement_action(
    observation: np.ndarray,
    env: VietnamStockEnv,
    step: int,
    agents: Sequence[PredictableAgent],
    policy: RegimeDisagreementPolicy,
) -> ActionDecision:
    """Apply disagreement-controlled rebalancing plus a VNINDEX downside cap."""

    policy.validate()
    intervals = {
        int(getattr(getattr(agent, "config", None), "rebalance_interval", 1))
        for agent in agents
    }
    rebalance_interval = intervals.pop() if len(intervals) == 1 else 1
    base_decision = _ensemble_action(
        observation,
        env,
        step,
        agents,
        policy.disagreement_policy(),
    )
    action, diagnostics = base_decision
    diagnostics = dict(diagnostics)
    if step % max(rebalance_interval, 1):
        diagnostics.update(
            {
                "market_trend_return": float("nan"),
                "market_risk_off": float("nan"),
                "regime_stock_exposure_cap": float("nan"),
                "pre_cap_stock_exposure": float("nan"),
                "post_cap_stock_exposure": float("nan"),
            }
        )
        return action, diagnostics

    end = env.current_index - 1
    start = end - policy.trend_lookback_days
    market_return = float("nan")
    if start >= 0:
        start_price = float(env.data.market_close[start])
        end_price = float(env.data.market_close[end])
        if (
            np.isfinite(start_price)
            and np.isfinite(end_price)
            and start_price > 0
            and end_price > 0
        ):
            market_return = end_price / start_price - 1.0
    risk_off = bool(np.isfinite(market_return) and market_return < 0.0)
    stock_cap = policy.risk_off_stock_exposure if risk_off else 1.0
    target = env.action_to_weights(action)
    pre_cap_exposure = float(target[:-1].sum())
    target = cap_stock_exposure(target, stock_cap)
    diagnostics.update(
        {
            "market_trend_return": market_return,
            "market_risk_off": float(risk_off),
            "regime_stock_exposure_cap": float(stock_cap),
            "pre_cap_stock_exposure": pre_cap_exposure,
            "post_cap_stock_exposure": float(target[:-1].sum()),
        }
    )
    return env.weights_to_action(target), diagnostics


def _record(info: dict[str, Any], tickers: tuple[str, ...]) -> dict[str, Any]:
    arrays = {
        "target_weights",
        "realized_weights",
        "available_shares",
        "pending_buy_shares",
        "buy_shares",
        "sell_shares",
    }
    row = {key: value for key, value in info.items() if key not in arrays}
    for index, ticker in enumerate(tickers):
        row[f"target_{ticker}"] = info["target_weights"][index]
        row[f"weight_{ticker}"] = info["realized_weights"][index]
        row[f"available_{ticker}"] = info["available_shares"][index]
        row[f"pending_buy_{ticker}"] = info["pending_buy_shares"][index]
        row[f"bought_{ticker}"] = info["buy_shares"][index]
        row[f"sold_{ticker}"] = info["sell_shares"][index]
    row["target_CASH"] = info["target_weights"][-1]
    row["weight_CASH"] = info["realized_weights"][-1]
    return row


def run_backtest(
    env: VietnamStockEnv,
    action_function: ActionFunction,
) -> pd.DataFrame:
    observation = env.reset()
    records: list[dict[str, Any]] = []
    step_index = 0
    done = False
    while not done:
        decision = action_function(observation, env, step_index)
        if isinstance(decision, tuple):
            action, diagnostics = decision
        else:
            action, diagnostics = decision, {}
        observation, _, done, info = env.step(action)
        row = _record(info, env.data.tickers)
        duplicate_keys = row.keys() & diagnostics.keys()
        if duplicate_keys:
            raise ValueError(
                f"Action diagnostics duplicate environment fields: {duplicate_keys}"
            )
        row.update(diagnostics)
        records.append(row)
        step_index += 1
    result = pd.DataFrame(records)
    result["date"] = pd.to_datetime(result["date"])
    return result


def evaluate_agent(
    env: VietnamStockEnv,
    agent: PredictableAgent,
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rebalance_interval = int(
        getattr(getattr(agent, "config", None), "rebalance_interval", 1)
    )

    def policy(
        observation: np.ndarray, current_env: VietnamStockEnv, step: int
    ) -> np.ndarray | None:
        if step % rebalance_interval:
            return None
        return agent.predict(observation, deterministic=True)

    frame = run_backtest(
        env,
        policy,
    )
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_mean_ensemble(
    env: VietnamStockEnv,
    agents: Sequence[PredictableAgent],
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(agents) < 2:
        raise ValueError("mean ensemble requires at least two agents")
    frame = run_backtest(
        env,
        lambda observation, current_env, step: _ensemble_action(
            observation, current_env, step, agents, None
        ),
    )
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_adaptive_rebalance_ensemble(
    env: VietnamStockEnv,
    agents: Sequence[PredictableAgent],
    policy: AdaptiveRebalancePolicy,
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(agents) < 2:
        raise ValueError("adaptive ensemble requires at least two agents")
    policy.validate()
    frame = run_backtest(
        env,
        lambda observation, current_env, step: _adaptive_rebalance_action(
            observation, current_env, step, agents, policy
        ),
    )
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_disagreement_ensemble(
    env: VietnamStockEnv,
    agents: Sequence[PredictableAgent],
    policy: DisagreementPolicy,
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(agents) < 2:
        raise ValueError("disagreement ensemble requires at least two agents")
    policy.validate()
    frame = run_backtest(
        env,
        lambda observation, current_env, step: _ensemble_action(
            observation,
            current_env,
            step,
            agents,
            policy,
        ),
    )
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_reliability_weighted_ensemble(
    env: VietnamStockEnv,
    agents: Sequence[PredictableAgent],
    policy: ReliabilityWeightedPolicy,
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(agents) != len(policy.seed_weights):
        raise ValueError("reliability policy must contain one weight per agent")
    policy.validate()
    weights = np.asarray(policy.seed_weights, dtype=np.float64)
    frame = run_backtest(
        env,
        lambda observation, current_env, step: _ensemble_action(
            observation,
            current_env,
            step,
            agents,
            policy.gate_policy,
            weights,
        ),
    )
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_directional_disagreement_ensemble(
    env: VietnamStockEnv,
    agents: Sequence[PredictableAgent],
    policy: DirectionalDisagreementPolicy,
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(agents) < 2:
        raise ValueError(
            "directional disagreement ensemble requires at least two agents"
        )
    policy.validate()
    frame = run_backtest(
        env,
        lambda observation, current_env, step: _directional_disagreement_action(
            observation, current_env, step, agents, policy
        ),
    )
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_regime_disagreement_ensemble(
    env: VietnamStockEnv,
    agents: Sequence[PredictableAgent],
    policy: RegimeDisagreementPolicy,
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(agents) < 2:
        raise ValueError("regime disagreement ensemble requires at least two agents")
    policy.validate()
    frame = run_backtest(
        env,
        lambda observation, current_env, step: _regime_disagreement_action(
            observation, current_env, step, agents, policy
        ),
    )
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_optimized_disagreement_ensemble(
    env: VietnamStockEnv,
    agents: Sequence[PredictableAgent],
    policy: DisagreementOptimizationPolicy,
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(agents) < 2:
        raise ValueError("disagreement optimization requires at least two agents")
    policy.validate()
    frame = run_backtest(
        env,
        lambda observation, current_env, step: _optimized_disagreement_action(
            observation, current_env, step, agents, policy
        ),
    )
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_equal_weight(
    env: VietnamStockEnv,
    *,
    rebalance_every: int = 1,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")

    def policy(_observation: np.ndarray, current_env: VietnamStockEnv, step: int):
        if step % rebalance_every:
            return None
        mask = current_env.current_tradable_mask
        weights = np.zeros(current_env.action_dim, dtype=np.float64)
        if mask.any():
            weights[:-1][mask] = 1.0 / mask.sum()
        else:
            weights[-1] = 1.0
        return current_env.weights_to_action(weights)

    frame = run_backtest(env, policy)
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_equal_weight_buy_and_hold(
    env: VietnamStockEnv,
    *,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    def policy(_observation: np.ndarray, current_env: VietnamStockEnv, step: int):
        if step > 0:
            return None
        mask = current_env.current_tradable_mask
        weights = np.zeros(current_env.action_dim, dtype=np.float64)
        if mask.any():
            weights[:-1][mask] = 1.0 / mask.sum()
        else:
            weights[-1] = 1.0
        return current_env.weights_to_action(weights)

    frame = run_backtest(env, policy)
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_cross_sectional_momentum(
    env: VietnamStockEnv,
    *,
    lookback_days: int = 60,
    top_k: int = 15,
    rebalance_every: int = 20,
    cash_weight: float = 0.05,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate a transparent momentum baseline under identical market rules.

    The portfolio ranks the last fully observed close-to-close returns, holds
    equal weights in the strongest names, and preserves a fixed cash buffer.
    It is aligned with the PPO momentum prior so the learned residual and
    disagreement gate are compared against a fair baseline.
    """

    if lookback_days not in (5, 20, 60):
        raise ValueError("lookback_days must be one of 5, 20, or 60")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    if not 0 <= cash_weight < 1:
        raise ValueError("cash_weight must be in [0, 1)")
    feature_index = STOCK_FEATURE_NAMES.index(f"return_{lookback_days}d")

    def policy(_observation: np.ndarray, current_env: VietnamStockEnv, step: int):
        if step % rebalance_every:
            return None
        # Observation windows end at current_index - 1: no execution-day close
        # or future information is used for the ranking.
        momentum = current_env.data.stock_features[
            current_env.current_index - 1, :, feature_index
        ]
        tradable = current_env.current_tradable_mask & np.isfinite(momentum)
        weights = np.zeros(current_env.action_dim, dtype=np.float64)
        candidates = np.flatnonzero(tradable)
        if candidates.size:
            selected_count = min(top_k, candidates.size)
            selected = candidates[
                np.argsort(momentum[candidates], kind="stable")[-selected_count:]
            ]
            weights[selected] = (1.0 - cash_weight) / selected_count
            weights[-1] = cash_weight
        else:
            weights[-1] = 1.0
        return current_env.weights_to_action(weights)

    frame = run_backtest(env, policy)
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def evaluate_inverse_volatility(
    env: VietnamStockEnv,
    *,
    lookback_days: int = 60,
    rebalance_every: int = 20,
    cash_weight: float = 0.05,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Monthly inverse-volatility baseline using completed observations only."""

    if lookback_days < 20:
        raise ValueError("lookback_days must be at least 20")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    if not 0 <= cash_weight < 1:
        raise ValueError("cash_weight must be in [0, 1)")

    def policy(_observation: np.ndarray, current_env: VietnamStockEnv, step: int):
        if step % rebalance_every:
            return None
        end = current_env.current_index
        start = max(0, end - lookback_days - 1)
        prices = current_env.data.close_prices[start:end]
        weights = np.zeros(current_env.action_dim, dtype=np.float64)
        if len(prices) < 2:
            weights[-1] = 1.0
            return current_env.weights_to_action(weights)
        valid_prices = np.isfinite(prices) & (prices > 0)
        log_prices = np.log(np.where(valid_prices, prices, np.nan))
        returns = np.diff(log_prices, axis=0)
        finite = np.isfinite(returns)
        observations = finite.sum(axis=0)
        means = np.divide(
            np.nansum(returns, axis=0),
            observations,
            out=np.zeros(returns.shape[1], dtype=np.float64),
            where=observations > 0,
        )
        squared = np.where(finite, (returns - means[None, :]) ** 2, 0.0).sum(
            axis=0
        )
        variance = np.divide(
            squared,
            observations - 1,
            out=np.full(returns.shape[1], np.nan, dtype=np.float64),
            where=observations > 1,
        )
        volatility = np.sqrt(variance)
        valid = (
            current_env.current_tradable_mask
            & np.isfinite(volatility)
            & (volatility > 1e-8)
            & (observations >= lookback_days // 2)
        )
        if valid.any():
            inverse = 1.0 / volatility[valid]
            weights[:-1][valid] = (1.0 - cash_weight) * inverse / inverse.sum()
            weights[-1] = cash_weight
        else:
            weights[-1] = 1.0
        return current_env.weights_to_action(weights)

    frame = run_backtest(env, policy)
    return frame, performance_metrics(
        frame,
        initial_cash=env.config.initial_cash,
        annual_risk_free_rate=annual_risk_free_rate,
    )


def paired_circular_block_bootstrap(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    *,
    block_length: int = 20,
    replications: int = 20_000,
    seed: int = 20_260_824,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Estimate a paired active-return interval while preserving local dependence."""

    strategy = np.asarray(strategy_returns, dtype=np.float64)
    baseline = np.asarray(baseline_returns, dtype=np.float64)
    if strategy.ndim != 1 or strategy.shape != baseline.shape or strategy.size < 2:
        raise ValueError("return arrays must be one-dimensional and equally sized")
    if not np.isfinite(strategy).all() or not np.isfinite(baseline).all():
        raise ValueError("returns must be finite")
    if block_length <= 0 or replications <= 0:
        raise ValueError("block_length and replications must be positive")

    active = strategy - baseline
    n_observations = active.size
    blocks_per_sample = int(np.ceil(n_observations / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, n_observations, size=(replications, blocks_per_sample)
    )
    indices = (starts[:, :, None] + np.arange(block_length)) % n_observations
    indices = indices.reshape(replications, -1)[:, :n_observations]
    bootstrap_means = active[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    observed_mean = float(active.mean())
    null_means = (active - observed_mean)[indices].mean(axis=1)
    p_value = (
        np.count_nonzero(np.abs(null_means) >= abs(observed_mean)) + 1
    ) / (replications + 1)
    return {
        "observations": float(n_observations),
        "block_length": float(block_length),
        "replications": float(replications),
        "annualized_mean_active_return": observed_mean * periods_per_year,
        "ci_95_lower": float(lower * periods_per_year),
        "ci_95_upper": float(upper * periods_per_year),
        "two_sided_p_value": float(p_value),
    }


def paired_hac_mean_test(
    strategy_returns: np.ndarray,
    baseline_returns: np.ndarray,
    *,
    max_lag: int = 20,
    periods_per_year: int = 252,
    noninferiority_margin_annual: float = 0.01,
) -> dict[str, float]:
    """HAC superiority test plus a pre-declared annual non-inferiority test."""

    strategy = np.asarray(strategy_returns, dtype=np.float64)
    baseline = np.asarray(baseline_returns, dtype=np.float64)
    if strategy.ndim != 1 or strategy.shape != baseline.shape or strategy.size < 2:
        raise ValueError("return arrays must be one-dimensional and equally sized")
    if not np.isfinite(strategy).all() or not np.isfinite(baseline).all():
        raise ValueError("returns must be finite")
    if max_lag < 0:
        raise ValueError("max_lag cannot be negative")
    if noninferiority_margin_annual < 0:
        raise ValueError("noninferiority margin cannot be negative")
    active = strategy - baseline
    centered = active - active.mean()
    n_observations = active.size
    lag = min(max_lag, n_observations - 1)
    long_run_variance = float(np.dot(centered, centered) / n_observations)
    for offset in range(1, lag + 1):
        autocovariance = float(
            np.dot(centered[offset:], centered[:-offset]) / n_observations
        )
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * autocovariance
    variance_of_mean = max(long_run_variance, 0.0) / n_observations
    standard_error = sqrt(variance_of_mean)
    mean = float(active.mean())
    if standard_error <= 1e-15:
        statistic = float("inf") if abs(mean) > 1e-15 else 0.0
        p_value = 0.0 if abs(mean) > 1e-15 else 1.0
    else:
        statistic = mean / standard_error
        p_value = erfc(abs(statistic) / sqrt(2.0))
    noninferiority_distance = (
        mean + noninferiority_margin_annual / periods_per_year
    )
    if standard_error <= 1e-15:
        if noninferiority_distance > 1e-15:
            noninferiority_statistic = float("inf")
            noninferiority_p_value = 0.0
        elif noninferiority_distance < -1e-15:
            noninferiority_statistic = float("-inf")
            noninferiority_p_value = 1.0
        else:
            noninferiority_statistic = 0.0
            noninferiority_p_value = 0.5
    else:
        noninferiority_statistic = (
            noninferiority_distance / standard_error
        )
        noninferiority_p_value = 0.5 * erfc(
            noninferiority_statistic / sqrt(2.0)
        )
    margin = 1.959963984540054 * standard_error * periods_per_year
    annualized = mean * periods_per_year
    return {
        "observations": float(n_observations),
        "max_lag": float(lag),
        "annualized_mean_active_return": annualized,
        "hac_standard_error_annualized": standard_error * periods_per_year,
        "hac_z_statistic": statistic,
        "hac_ci_95_lower": annualized - margin,
        "hac_ci_95_upper": annualized + margin,
        "hac_two_sided_p_value": p_value,
        "noninferiority_margin_annual": noninferiority_margin_annual,
        "hac_noninferiority_z_statistic": noninferiority_statistic,
        "hac_noninferiority_p_value": noninferiority_p_value,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def performance_metrics(
    frame: pd.DataFrame,
    *,
    initial_cash: float,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute return, risk, benchmark, execution and cost diagnostics."""

    if frame.empty:
        raise ValueError("Cannot evaluate an empty backtest")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Backtest dates must be unique and chronological")
    returns = frame["daily_return"].to_numpy(dtype=np.float64)
    benchmark = frame["benchmark_return"].to_numpy(dtype=np.float64)
    reported_nav = frame["nav"].to_numpy(dtype=np.float64)
    if (
        initial_cash <= 0
        or not np.isfinite(returns).all()
        or not np.isfinite(benchmark).all()
        or not np.isfinite(reported_nav).all()
        or (returns <= -1.0).any()
        or (benchmark <= -1.0).any()
        or (reported_nav <= 0.0).any()
    ):
        raise ValueError("Backtest returns and NAV must be finite and positive")
    expected_nav = initial_cash * np.cumprod(1.0 + returns)
    if not np.allclose(
        reported_nav,
        expected_nav,
        rtol=1e-8,
        atol=max(1e-6, initial_cash * 1e-10),
    ):
        raise ValueError("Backtest NAV is inconsistent with compounded daily returns")
    nav = np.concatenate([[initial_cash], reported_nav])
    elapsed_days = max((dates.iloc[-1] - dates.iloc[0]).days, 1)
    # CAGR follows elapsed calendar time. Risk/turnover metrics below use the
    # explicitly separate trading-session convention ``periods_per_year``.
    # Keeping these clocks separate avoids the ambiguous 250-vs-252 inference
    # that arises when CAGR is backed out from a session count.
    years = elapsed_days / 365.25
    total_return = nav[-1] / initial_cash - 1.0
    cagr = (nav[-1] / initial_cash) ** (1.0 / years) - 1.0
    risk_free_daily = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - risk_free_daily
    volatility = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
    sharpe = _safe_ratio(
        float(np.mean(excess) * periods_per_year), volatility
    )
    downside = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year))
    sortino = _safe_ratio(
        float(np.mean(excess) * periods_per_year), downside_deviation
    )
    running_peak = np.maximum.accumulate(nav)
    drawdowns = nav / running_peak - 1.0
    maximum_drawdown = float(drawdowns.min())
    calmar = _safe_ratio(cagr, abs(maximum_drawdown))
    var_95 = float(-np.quantile(returns, 0.05))
    tail = returns[returns <= np.quantile(returns, 0.05)]
    cvar_95 = float(-tail.mean()) if len(tail) else float("nan")

    benchmark_nav = np.cumprod(np.concatenate([[1.0], 1.0 + benchmark]))
    benchmark_total_return = float(benchmark_nav[-1] - 1.0)
    benchmark_cagr = float(benchmark_nav[-1] ** (1.0 / years) - 1.0)
    benchmark_volatility = float(
        np.std(benchmark, ddof=1) * np.sqrt(periods_per_year)
    )
    benchmark_sharpe = _safe_ratio(
        float(np.mean(benchmark - risk_free_daily) * periods_per_year),
        benchmark_volatility,
    )
    benchmark_downside = np.minimum(benchmark - risk_free_daily, 0.0)
    benchmark_downside_deviation = float(
        np.sqrt(np.mean(benchmark_downside**2)) * np.sqrt(periods_per_year)
    )
    benchmark_sortino = _safe_ratio(
        float(np.mean(benchmark - risk_free_daily) * periods_per_year),
        benchmark_downside_deviation,
    )
    benchmark_peak = np.maximum.accumulate(benchmark_nav)
    benchmark_mdd = float((benchmark_nav / benchmark_peak - 1.0).min())
    active_returns = returns - benchmark
    tracking_error = float(np.std(active_returns, ddof=1) * np.sqrt(periods_per_year))
    information_ratio = _safe_ratio(
        float(np.mean(active_returns) * periods_per_year), tracking_error
    )
    benchmark_variance = float(np.var(benchmark, ddof=1))
    beta = (
        float(np.cov(returns, benchmark, ddof=1)[0, 1] / benchmark_variance)
        if benchmark_variance > 1e-12
        else float("nan")
    )
    alpha_annual = (
        float(
            (
                np.mean(returns - risk_free_daily)
                - beta * np.mean(benchmark - risk_free_daily)
            )
            * periods_per_year
        )
        if np.isfinite(beta)
        else float("nan")
    )

    broker_fees = float(frame["broker_fees"].sum())
    sell_tax = float(frame["sell_tax"].sum())
    slippage_cost = (
        float(frame["slippage_cost"].sum())
        if "slippage_cost" in frame
        else 0.0
    )
    total_transaction_cost = broker_fees + sell_tax + slippage_cost
    stock_weight_columns = [
        column
        for column in frame.columns
        if column.startswith("weight_") and column != "weight_CASH"
    ]
    if stock_weight_columns:
        stock_weights = frame[stock_weight_columns].to_numpy(dtype=np.float64)
        stock_weight_sums = stock_weights.sum(axis=1, keepdims=True)
        normalized_stock_weights = np.divide(
            stock_weights,
            stock_weight_sums,
            out=np.zeros_like(stock_weights),
            where=stock_weight_sums > 1e-12,
        )
        concentration = np.sum(normalized_stock_weights**2, axis=1)
        effective_assets = np.divide(
            1.0,
            concentration,
            out=np.zeros_like(concentration),
            where=concentration > 1e-12,
        )
        mean_max_stock_weight = float(np.max(stock_weights, axis=1).mean())
        mean_effective_assets = float(effective_assets.mean())
    else:
        mean_max_stock_weight = float("nan")
        mean_effective_assets = float("nan")
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "cagr_elapsed_calendar_days": float(elapsed_days),
        "cagr_elapsed_years": float(years),
        "annualization_periods_per_year": float(periods_per_year),
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": calmar,
        "var_95_daily": var_95,
        "cvar_95_daily": cvar_95,
        "positive_day_rate": float(np.mean(returns > 0)),
        "benchmark_total_return": benchmark_total_return,
        "benchmark_cagr": benchmark_cagr,
        "benchmark_annual_volatility": benchmark_volatility,
        "benchmark_sharpe": benchmark_sharpe,
        "benchmark_sortino": benchmark_sortino,
        "benchmark_maximum_drawdown": benchmark_mdd,
        "excess_total_return": float(total_return - benchmark_total_return),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha_annual": alpha_annual,
        "total_turnover": float(frame["turnover"].sum()),
        "annual_turnover": float(frame["turnover"].mean() * periods_per_year),
        "annual_turnover_percent_nav": float(
            frame["turnover"].mean() * periods_per_year * 100.0
        ),
        "annual_risk_free_rate": float(annual_risk_free_rate),
        "trade_count": float(frame["trade_count"].sum()),
        "broker_fees": broker_fees,
        "sell_tax": sell_tax,
        "slippage_cost": slippage_cost,
        "total_transaction_cost": total_transaction_cost,
        "transaction_cost_to_initial": float(
            total_transaction_cost / initial_cash
        ),
        "mean_available_cash_weight": float(
            (frame["available_cash"] / frame["nav"]).mean()
        ),
        "mean_pending_cash_weight": float(
            (frame["pending_sale_cash"] / frame["nav"]).mean()
        ),
        "mean_max_stock_weight": mean_max_stock_weight,
        "mean_effective_assets": mean_effective_assets,
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).T.rename_axis("strategy")


def save_equity_plot(
    histories: dict[str, pd.DataFrame],
    path: str | Path,
    *,
    title: str = "Out-of-sample portfolio value",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 6))
    benchmark_drawn = False
    for name, frame in histories.items():
        normalized = frame["nav"] / frame["pretrade_nav"].iloc[0]
        axis.plot(frame["date"], normalized, label=name, linewidth=1.5)
        if not benchmark_drawn:
            benchmark_curve = (1.0 + frame["benchmark_return"]).cumprod()
            axis.plot(
                frame["date"],
                benchmark_curve,
                label="VNINDEX",
                color="black",
                linestyle="--",
                linewidth=1.2,
            )
            benchmark_drawn = True
    axis.set_title(title)
    axis.set_ylabel("Normalized value")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
