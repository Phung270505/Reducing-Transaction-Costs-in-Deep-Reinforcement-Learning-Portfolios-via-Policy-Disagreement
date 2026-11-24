from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vn30ppo.config import PPOConfig
from vn30ppo.environment import ObservationSpec
from vn30ppo.ppo import (
    PPOAgent,
    TRAINING_ALGORITHM_VERSION,
    _validation_checkpoint_score,
)
from tests.test_environment import make_env


def _swap_assets_in_observation(
    observation: np.ndarray,
    spec: ObservationSpec,
    first: int,
    second: int,
) -> np.ndarray:
    swapped = observation.copy()
    asset_history = swapped[: spec.asset_history_size].reshape(
        spec.n_assets, spec.lookback, spec.n_stock_features
    )
    asset_history[[first, second]] = asset_history[[second, first]]
    portfolio_offset = spec.asset_history_size + spec.market_history_size
    portfolio = swapped[
        portfolio_offset : portfolio_offset
        + spec.n_assets * spec.per_asset_portfolio_features
    ].reshape(spec.n_assets, spec.per_asset_portfolio_features)
    portfolio[[first, second]] = portfolio[[second, first]]
    return swapped


def test_attention_policy_produces_finite_stock_and_cash_actions() -> None:
    spec = ObservationSpec(
        n_assets=3,
        lookback=4,
        n_stock_features=5,
        n_market_features=4,
        settlement_days=2,
    )
    config = PPOConfig(
        total_timesteps=8,
        rollout_steps=8,
        batch_size=8,
        update_epochs=1,
        asset_hidden_size=16,
        context_hidden_size=32,
        attention_heads=4,
        attention_layers=1,
        seed=5,
    )
    agent = PPOAgent(spec, config)
    observation = np.zeros(spec.observation_dim, dtype=np.float32)
    portfolio_offset = spec.asset_history_size + spec.market_history_size
    portfolio = observation[
        portfolio_offset : portfolio_offset
        + spec.n_assets * spec.per_asset_portfolio_features
    ].reshape(spec.n_assets, spec.per_asset_portfolio_features)
    portfolio[:, 0] = 1.0
    action = agent.predict(observation)
    assert action.shape == (spec.action_dim,)
    assert np.isfinite(action).all()


def test_policy_is_permutation_equivariant_across_stock_slots() -> None:
    spec = ObservationSpec(
        n_assets=3,
        lookback=4,
        n_stock_features=5,
        n_market_features=4,
        settlement_days=2,
    )
    config = PPOConfig(
        total_timesteps=8,
        rollout_steps=8,
        batch_size=8,
        update_epochs=1,
        asset_hidden_size=16,
        context_hidden_size=32,
        attention_heads=4,
        attention_layers=1,
        momentum_prior_topk=0,
        seed=5,
    )
    agent = PPOAgent(spec, config)
    rng = np.random.default_rng(17)
    observation = rng.normal(size=spec.observation_dim).astype(np.float32)
    portfolio_offset = spec.asset_history_size + spec.market_history_size
    portfolio = observation[
        portfolio_offset : portfolio_offset
        + spec.n_assets * spec.per_asset_portfolio_features
    ].reshape(spec.n_assets, spec.per_asset_portfolio_features)
    portfolio[:, 0] = 1.0
    swapped = _swap_assets_in_observation(observation, spec, 0, 1)

    agent.model.eval()
    with torch.no_grad():
        original_means, original_value, _ = agent.model(
            torch.as_tensor(observation, device=agent.device)
        )
        swapped_means, swapped_value, _ = agent.model(
            torch.as_tensor(swapped, device=agent.device)
        )

    expected_means = original_means.clone()
    expected_means[:, [0, 1]] = expected_means[:, [1, 0]]
    assert torch.allclose(swapped_means, expected_means, atol=1e-6, rtol=1e-6)
    assert torch.allclose(swapped_value, original_value, atol=1e-6, rtol=1e-6)


def test_momentum_prior_tilts_policy_to_top_ranked_assets() -> None:
    spec = ObservationSpec(
        n_assets=4,
        lookback=3,
        n_stock_features=5,
        n_market_features=4,
        settlement_days=2,
    )
    config = PPOConfig(
        total_timesteps=8,
        rollout_steps=8,
        batch_size=8,
        update_epochs=1,
        asset_hidden_size=16,
        context_hidden_size=32,
        attention_heads=4,
        attention_layers=1,
        momentum_prior_topk=2,
        momentum_prior_cash_weight=0.05,
        momentum_prior_strength=1.0,
        seed=5,
    )
    agent = PPOAgent(spec, config)
    observation = np.zeros(spec.observation_dim, dtype=np.float32)
    asset_history = observation[: spec.asset_history_size].reshape(
        spec.n_assets, spec.lookback, spec.n_stock_features
    )
    asset_history[:, -1, 3] = [-1.0, 2.0, 0.5, 3.0]
    portfolio_offset = spec.asset_history_size + spec.market_history_size
    portfolio = observation[
        portfolio_offset : portfolio_offset
        + spec.n_assets * spec.per_asset_portfolio_features
    ].reshape(spec.n_assets, spec.per_asset_portfolio_features)
    portfolio[:, 0] = 1.0
    logits = agent.predict(observation)
    probabilities = np.exp(logits - logits.max())
    weights = probabilities / probabilities.sum()
    assert set(np.argsort(weights[:-1])[-2:]) == {1, 3}
    assert weights[[1, 3]].sum() > 0.90


def test_rollout_uses_market_day_discounts_for_held_decisions() -> None:
    env = make_env(n_assets=2)
    config = PPOConfig(
        total_timesteps=1,
        rollout_steps=1,
        batch_size=1,
        update_epochs=1,
        asset_hidden_size=16,
        context_hidden_size=32,
        attention_heads=4,
        attention_layers=1,
        momentum_prior_topk=0,
        rebalance_interval=3,
        gamma=0.90,
        gae_lambda=0.80,
        device="cpu",
        seed=5,
    )
    agent = PPOAgent(env.spec, config)
    batch, _, _ = agent._collect_rollout(env, env.reset(), steps=1)
    assert np.isclose(batch.transition_discounts[0], 0.90**3)
    assert np.isclose(batch.trace_discounts[0], (0.90 * 0.80) ** 3)


def test_profit_checkpoint_score_handles_zero_volatility_cash_policy() -> None:
    config = PPOConfig(validation_objective="profit")
    score = _validation_checkpoint_score(
        {
            "cagr": 0.0,
            "sharpe_ratio": float("nan"),
            "annual_turnover": 0.0,
            "maximum_drawdown": 0.0,
        },
        config,
    )
    assert score == 0.0


def test_checkpoint_rejects_training_algorithm_version_mismatch(
    tmp_path: Path,
) -> None:
    spec = ObservationSpec(
        n_assets=2,
        lookback=3,
        n_stock_features=5,
        n_market_features=4,
        settlement_days=2,
    )
    agent = PPOAgent(
        spec,
        PPOConfig(
            total_timesteps=8,
            rollout_steps=8,
            batch_size=8,
            update_epochs=1,
            asset_hidden_size=16,
            context_hidden_size=32,
            attention_heads=4,
            attention_layers=1,
            seed=5,
        ),
    )
    path = tmp_path / "checkpoint.pt"
    agent.save(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["training_algorithm_version"] = TRAINING_ALGORITHM_VERSION - 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Checkpoint version mismatch"):
        PPOAgent.load(path, device="cpu")
