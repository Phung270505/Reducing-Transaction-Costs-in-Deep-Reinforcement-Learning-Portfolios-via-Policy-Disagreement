from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import isfinite
from typing import Any


@dataclass(slots=True)
class MarketConfig:
    """Execution and accounting assumptions used by the simulator.

    Prices returned by vnstock are expressed in thousands of VND, hence the
    default ``price_multiplier``. Every friction is configurable so an
    experiment can report sensitivity instead of hiding assumptions.
    """

    initial_cash: float = 1_000_000_000.0
    settlement_days: int = 2
    price_multiplier: float = 1_000.0
    lot_size: int = 100
    legacy_lot_size: int = 10
    lot_size_100_effective_date: str = "2021-01-04"
    odd_lot_trading_effective_date: str = "2022-09-12"
    broker_fee_rate: float = 0.0015
    sell_tax_rate: float = 0.001
    slippage_bps: float = 5.0
    max_volume_fraction: float = 0.01
    price_limit_rate: float = 0.07
    reward_scale: float = 100.0
    turnover_penalty_rate: float = 0.0005
    drawdown_penalty: float = 0.01
    minimum_trade_fraction: float = 0.005
    allocation_alpha_reward: float = 1.0

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.settlement_days < 0:
            raise ValueError("settlement_days cannot be negative")
        if (
            self.price_multiplier <= 0
            or self.lot_size <= 0
            or self.legacy_lot_size <= 0
        ):
            raise ValueError("price_multiplier and lot sizes must be positive")
        lot_change = date.fromisoformat(self.lot_size_100_effective_date)
        odd_lot_start = date.fromisoformat(self.odd_lot_trading_effective_date)
        if odd_lot_start < lot_change:
            raise ValueError(
                "odd-lot trading cannot predate the 100-share lot regime"
            )
        if not 0 <= self.slippage_bps <= 10_000:
            raise ValueError("slippage_bps must be in [0, 10000]")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        for name in (
            "broker_fee_rate",
            "sell_tax_rate",
            "max_volume_fraction",
            "price_limit_rate",
            "turnover_penalty_rate",
            "drawdown_penalty",
            "minimum_trade_fraction",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.allocation_alpha_reward <= 10:
            raise ValueError("allocation_alpha_reward must be in [0, 10]")


@dataclass(slots=True)
class PPOConfig:
    total_timesteps: int = 200_000
    rollout_steps: int = 512
    update_epochs: int = 6
    batch_size: int = 128
    learning_rate: float = 1.5e-4
    final_learning_rate_fraction: float = 0.10
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_range: float = 0.20
    value_clip_range: float = 0.20
    entropy_coef: float = 0.0005
    final_entropy_coef: float = 0.0
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    target_kl: float = 0.02
    asset_hidden_size: int = 64
    context_hidden_size: int = 192
    architecture: str = "attention"
    attention_heads: int = 4
    attention_layers: int = 2
    dropout: float = 0.0
    momentum_prior_topk: int = 15
    momentum_prior_cash_weight: float = 0.05
    momentum_prior_strength: float = 0.25
    normalize_entropy_by_action_dim: bool = True
    penalty_warmup_fraction: float = 0.25
    validation_objective: str = "profit"
    rebalance_interval: int = 20
    validation_turnover_penalty: float = 0.005
    validation_drawdown_penalty: float = 0.10
    seed: int = 42
    device: str = "auto"

    def validate(self) -> None:
        if self.total_timesteps <= 0 or self.rollout_steps <= 0:
            raise ValueError("total_timesteps and rollout_steps must be positive")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if self.batch_size <= 0 or self.batch_size > self.rollout_steps:
            raise ValueError("batch_size must be in [1, rollout_steps]")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in (0, 1]")
        if not 0 < self.final_learning_rate_fraction <= 1:
            raise ValueError("final_learning_rate_fraction must be in (0, 1]")
        if self.final_entropy_coef < 0 or self.entropy_coef < 0:
            raise ValueError("entropy coefficients cannot be negative")
        if not 0 < self.clip_range <= 1 or not 0 < self.value_clip_range <= 1:
            raise ValueError("PPO clip ranges must be in (0, 1]")
        if self.value_coef < 0 or self.max_grad_norm <= 0 or self.target_kl <= 0:
            raise ValueError(
                "value_coef must be non-negative and gradient/KL limits positive"
            )
        if self.asset_hidden_size <= 0 or self.context_hidden_size < 2:
            raise ValueError("network hidden sizes are too small")
        if self.architecture not in {"legacy", "attention"}:
            raise ValueError("architecture must be 'legacy' or 'attention'")
        if self.attention_heads <= 0 or self.attention_layers <= 0:
            raise ValueError("attention_heads and attention_layers must be positive")
        if self.asset_hidden_size % self.attention_heads:
            raise ValueError("asset_hidden_size must be divisible by attention_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.momentum_prior_topk < 0:
            raise ValueError("momentum_prior_topk cannot be negative")
        if not 0 < self.momentum_prior_cash_weight < 1:
            raise ValueError("momentum_prior_cash_weight must be in (0, 1)")
        if not 0 <= self.momentum_prior_strength <= 1:
            raise ValueError("momentum_prior_strength must be in [0, 1]")
        if not 0 <= self.penalty_warmup_fraction <= 1:
            raise ValueError("penalty_warmup_fraction must be in [0, 1]")
        if self.validation_objective not in {"risk_adjusted", "profit"}:
            raise ValueError(
                "validation_objective must be 'risk_adjusted' or 'profit'"
            )
        if self.rebalance_interval <= 0:
            raise ValueError("rebalance_interval must be positive")
        if (
            self.validation_turnover_penalty < 0
            or self.validation_drawdown_penalty < 0
        ):
            raise ValueError("validation penalties cannot be negative")


@dataclass(slots=True)
class EnsembleConfig:
    """Validation-only search space for disagreement-aware rebalancing.

    Disagreement is the mean pairwise total-variation distance between the
    portfolio weights proposed by independently trained PPO policies. Gate
    thresholds and regularized seed-reliability weights are fitted only on the
    calibration period and never on the test period.
    """

    low_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)
    high_quantiles: tuple[float, ...] = (0.75, 0.90)
    minimum_rebalance_grid: tuple[float, ...] = (0.0, 0.10, 1.0)
    soft_disagreement_k_grid: tuple[float, ...] = (
        0.50,
        1.0,
        1.50,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
    )
    adaptive_turnover_budget_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    adaptive_turnover_k_grid: tuple[float, ...] = (1.0, 2.0, 4.0)
    adaptive_cost_penalty_grid: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02)
    adaptive_cost_k_grid: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    asset_disagreement_k_grid: tuple[float, ...] = (2.50, 5.0, 10.0, 20.0)
    hard_partial_rebalance_grid: tuple[float, ...] = (0.25, 0.50, 0.75)
    combined_k_grid: tuple[float, ...] = (0.50, 1.0, 2.0, 4.0, 8.0)
    combined_turnover_budget_grid: tuple[float, ...] = (1.0, 0.80, 0.70, 0.50, 0.35)
    combined_global_penalty_grid: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02)
    combined_asset_penalty_grid: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0)
    combined_consensus_methods: tuple[str, ...] = (
        "mean",
        "median",
        "geometric_median",
    )
    # Kept only so older checkpoint configurations remain loadable.
    combined_consensus_method: str = "geometric_median"
    adaptive_selection_tolerance: float = 0.002
    reliability_temperature_grid: tuple[float, ...] = (
        0.10,
        0.25,
        0.50,
        1.0,
        2.0,
    )
    directional_shrinkage_grid: tuple[float, ...] = (0.0, 0.50, 1.0, 2.0)
    directional_turnover_regularization_grid: tuple[float, ...] = (
        0.0,
        0.50,
        1.0,
    )
    # Any positive calibration improvement over the arithmetic mean is enough
    # to select the distinct D3R policy.
    directional_minimum_validation_improvement: float = 0.0
    regime_lookback_days: int = 60
    regime_risk_off_exposure_grid: tuple[float, ...] = (0.0, 0.50, 1.0)
    optimization_disagreement_coefficient_grid: tuple[float, ...] = (
        -8.0,
        -2.0,
        0.0,
        2.0,
        8.0,
    )
    optimization_turnover_regularization_grid: tuple[float, ...] = (
        0.0,
        0.50,
        1.0,
    )
    optimization_consensus_methods: tuple[str, ...] = (
        "mean",
        "median",
        "trimmed_mean",
        "geometric",
        "geometric_sparsemax",
    )
    optimization_primary_consensus_method: str = "mean"
    # Regime and alternative-consensus ablations remain opt-in. The three core
    # ensemble strategies are always evaluated.
    run_extended_ablations: bool = False

    def validate(self) -> None:
        if not isinstance(self.run_extended_ablations, bool):
            raise ValueError("run_extended_ablations must be boolean")
        if not self.low_quantiles or not self.high_quantiles:
            raise ValueError("disagreement quantile grids cannot be empty")
        for name in (
            "low_quantiles",
            "high_quantiles",
        ):
            values = getattr(self, name)
            if any(not 0 < value < 1 for value in values):
                raise ValueError(f"{name} values must be in (0, 1)")
        if not self.minimum_rebalance_grid:
            raise ValueError("minimum_rebalance_grid cannot be empty")
        if any(not 0 <= value <= 1 for value in self.minimum_rebalance_grid):
            raise ValueError("minimum rebalance values must be in [0, 1]")
        positive_grids = (
            "soft_disagreement_k_grid",
            "adaptive_turnover_budget_grid",
            "adaptive_turnover_k_grid",
            "adaptive_cost_penalty_grid",
            "adaptive_cost_k_grid",
            "asset_disagreement_k_grid",
            "hard_partial_rebalance_grid",
            "combined_asset_penalty_grid",
            "combined_k_grid",
            "combined_turnover_budget_grid",
        )
        for name in positive_grids:
            values = getattr(self, name)
            if not values or any(
                not isinstance(value, (int, float)) or value <= 0
                for value in values
            ):
                raise ValueError(f"{name} values must be positive")
        if any(value > 1 for value in self.adaptive_turnover_budget_grid):
            raise ValueError("adaptive turnover budgets must be at most one")
        if any(value > 1 for value in self.combined_turnover_budget_grid):
            raise ValueError("combined turnover budgets must be at most one")
        if any(value > 1 for value in self.hard_partial_rebalance_grid):
            raise ValueError("hard partial gates must be at most one")
        if (
            not self.combined_global_penalty_grid
            or any(value < 0 for value in self.combined_global_penalty_grid)
        ):
            raise ValueError("combined global penalties must be non-negative")
        combined_lengths = {
            len(self.combined_k_grid),
            len(self.combined_turnover_budget_grid),
            len(self.combined_global_penalty_grid),
            len(self.combined_asset_penalty_grid),
        }
        if len(combined_lengths) != 1:
            raise ValueError("combined profile grids must have matching lengths")
        allowed_attached_consensus = {"mean", "median", "geometric_median"}
        if (
            not self.combined_consensus_methods
            or any(
                method not in allowed_attached_consensus
                for method in self.combined_consensus_methods
            )
        ):
            raise ValueError("invalid combined consensus methods")
        if (
            not isfinite(self.adaptive_selection_tolerance)
            or self.adaptive_selection_tolerance < 0
        ):
            raise ValueError("adaptive selection tolerance must be non-negative")
        if (
            not self.reliability_temperature_grid
            or any(value <= 0 for value in self.reliability_temperature_grid)
        ):
            raise ValueError("reliability temperatures must be positive")
        if (
            not self.directional_shrinkage_grid
            or any(value < 0 for value in self.directional_shrinkage_grid)
        ):
            raise ValueError(
                "directional shrinkage coefficients must be non-negative"
            )
        if (
            not self.directional_turnover_regularization_grid
            or any(
                value < 0
                for value in self.directional_turnover_regularization_grid
            )
        ):
            raise ValueError(
                "directional turnover regularization must be non-negative"
            )
        if self.directional_minimum_validation_improvement < 0:
            raise ValueError(
                "directional minimum validation improvement must be "
                "non-negative"
            )
        if self.regime_lookback_days < 20:
            raise ValueError("regime lookback must be at least 20 trading days")
        if (
            not self.regime_risk_off_exposure_grid
            or any(
                not 0 <= value <= 1
                for value in self.regime_risk_off_exposure_grid
            )
        ):
            raise ValueError("risk-off exposures must be in [0, 1]")
        if not any(low < high for low in self.low_quantiles for high in self.high_quantiles):
            raise ValueError("at least one low quantile must be below a high quantile")
        if (
            not self.optimization_disagreement_coefficient_grid
            or any(
                not isinstance(value, (int, float))
                for value in self.optimization_disagreement_coefficient_grid
            )
        ):
            raise ValueError("disagreement coefficient grid must be numeric")
        if (
            not self.optimization_turnover_regularization_grid
            or any(
                value < 0
                for value in self.optimization_turnover_regularization_grid
            )
        ):
            raise ValueError(
                "optimization turnover regularization must be non-negative"
            )
        allowed_consensus = {
            "mean",
            "median",
            "trimmed_mean",
            "geometric",
            "geometric_sparsemax",
        }
        if (
            not self.optimization_consensus_methods
            or any(
                method not in allowed_consensus
                for method in self.optimization_consensus_methods
            )
        ):
            raise ValueError("invalid optimization consensus method")
        if (
            self.optimization_primary_consensus_method
            not in self.optimization_consensus_methods
        ):
            raise ValueError(
                "primary consensus method must be included in the ablation grid"
            )


@dataclass(slots=True)
class ExperimentConfig:
    data_path: str = "Data/VN30_VNINDEX_2016_2025.csv"
    universe_path: str = "Data/VN30_VNINDEX_2016_2025.csv"
    require_point_in_time_universe: bool = False
    expected_universe_size: int = 30
    train_start: str = "2016-01-01"
    train_end: str = "2021-12-31"
    validation_start: str = "2022-01-01"
    validation_end: str = "2022-12-31"
    calibration_start: str = "2023-01-01"
    calibration_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-12-31"
    test_initial_cash_delay_days: int = 0
    lookback: int = 20
    minimum_training_days: int = 252
    training_episode_days: int = 504
    output_dir: str = "artifacts"

    def validate(self) -> None:
        if self.require_point_in_time_universe and not self.universe_path.strip():
            raise ValueError(
                "universe_path is required when point-in-time membership is enforced"
            )
        if self.expected_universe_size <= 0:
            raise ValueError("expected_universe_size must be positive")
        if self.test_initial_cash_delay_days < 0:
            raise ValueError("test_initial_cash_delay_days cannot be negative")
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.minimum_training_days < self.lookback:
            raise ValueError("minimum_training_days must be >= lookback")
        if self.training_episode_days <= 0:
            raise ValueError("training_episode_days must be positive")
        boundaries = [
            date.fromisoformat(value)
            for value in (
                self.train_start,
                self.train_end,
                self.validation_start,
                self.validation_end,
                self.calibration_start,
                self.calibration_end,
                self.test_start,
                self.test_end,
            )
        ]
        valid_order = (
            boundaries[0] <= boundaries[1]
            < boundaries[2] <= boundaries[3]
            < boundaries[4] <= boundaries[5]
            < boundaries[6] <= boundaries[7]
        )
        if not valid_order:
            raise ValueError(
                "experiment periods must be chronological and non-overlapping"
            )


def dataclass_dict(value: Any) -> dict[str, Any]:
    """Return a JSON-friendly representation of a dataclass."""

    result = asdict(value)
    for key, item in result.items():
        if isinstance(item, tuple):
            result[key] = list(item)
    return result
