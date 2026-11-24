from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .config import PPOConfig, dataclass_dict
from .environment import ObservationSpec, VietnamStockEnv

MODEL_VERSION = 3
TRAINING_ALGORITHM_VERSION = 6


def _validation_checkpoint_score(
    metrics: dict[str, float], config: PPOConfig
) -> float:
    """Return a finite checkpoint score or ``-inf`` for invalid metrics."""

    cagr = float(metrics["cagr"])
    sharpe = float(metrics["sharpe_ratio"])
    turnover = float(metrics["annual_turnover"])
    drawdown = abs(float(metrics["maximum_drawdown"]))
    if not all(np.isfinite(value) for value in (cagr, turnover, drawdown)):
        return -np.inf
    if config.validation_objective == "profit":
        # A zero-volatility cash policy has an undefined Sharpe but a valid
        # return. Treat the undefined auxiliary term as neutral.
        sharpe_component = sharpe if np.isfinite(sharpe) else 0.0
        return float(
            cagr
            + 0.05 * sharpe_component
            - config.validation_turnover_penalty * turnover
            - config.validation_drawdown_penalty * drawdown
        )
    if not np.isfinite(sharpe):
        return -np.inf
    return float(
        sharpe
        - config.validation_turnover_penalty * turnover
        - config.validation_drawdown_penalty * drawdown
    )


def resolve_training_device(requested: str) -> torch.device:
    """Resolve a requested device, preferring CUDA and failing explicitly."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this Python interpreter. "
            f"PyTorch={torch.__version__}, CUDA build={torch.version.cuda}. "
            "Install the CUDA wheel and make sure PyCharm uses the project .venv."
        )
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    return device


def configure_training_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def training_device_info(requested: str = "auto") -> dict[str, Any]:
    device = resolve_training_device(requested)
    configure_training_device(device)
    info: dict[str, Any] = {
        "requested": requested,
        "selected": str(device),
        "type": device.type.upper(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        info.update(
            {
                "gpu_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "vram_gb": total_memory / 1024**3,
                "free_vram_gb": free_memory / 1024**3,
                "cudnn_version": torch.backends.cudnn.version(),
                "tf32_enabled": True,
            }
        )
    else:
        info["cpu_threads"] = torch.get_num_threads()
    return info


def format_device_banner(info: dict[str, Any]) -> str:
    lines = [
        "=== TRAINING DEVICE ===",
        f"Requested       : {info['requested']}",
        f"Selected        : {info['selected']} ({info['type']})",
        f"PyTorch         : {info['pytorch_version']}",
        f"CUDA available  : {info['cuda_available']}",
    ]
    if info["type"] == "CUDA":
        lines.extend(
            [
                f"GPU             : {info['gpu_name']}",
                f"CUDA runtime    : {info['cuda_runtime']}",
                f"cuDNN           : {info['cudnn_version']}",
                f"Compute cap.    : {info['compute_capability']}",
                f"VRAM            : {info['free_vram_gb']:.2f} GB free / "
                f"{info['vram_gb']:.2f} GB total",
                f"TF32 acceleration: {info['tf32_enabled']}",
            ]
        )
    else:
        lines.extend(
            [
                f"CPU threads     : {info['cpu_threads']}",
                "GPU acceleration: unavailable in this interpreter",
            ]
        )
    return "\n".join(lines)


def format_duration(seconds: float) -> str:
    """Format a duration for compact, stable console progress output."""

    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total_seconds = int(round(seconds))
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


def _layer_init(layer: nn.Linear, gain: float = math.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class PortfolioActorCritic(nn.Module):
    """Shared temporal encoder with stock-wise actor heads and a global critic."""

    def __init__(
        self,
        spec: ObservationSpec,
        *,
        asset_hidden_size: int = 32,
        context_hidden_size: int = 128,
        architecture: str = "attention",
        attention_heads: int = 4,
        attention_layers: int = 2,
        dropout: float = 0.0,
        momentum_prior_topk: int = 0,
        momentum_prior_cash_weight: float = 0.05,
        momentum_prior_strength: float = 1.0,
        normalize_entropy_by_action_dim: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.asset_hidden_size = asset_hidden_size
        self.context_hidden_size = context_hidden_size
        self.architecture = architecture
        self.momentum_prior_topk = momentum_prior_topk
        self.momentum_prior_cash_weight = momentum_prior_cash_weight
        self.momentum_prior_strength = momentum_prior_strength
        self.normalize_entropy_by_action_dim = normalize_entropy_by_action_dim
        self.asset_norm = nn.LayerNorm(spec.n_stock_features)
        self.market_norm = nn.LayerNorm(spec.n_market_features)
        self.asset_encoder = nn.GRU(
            spec.n_stock_features, asset_hidden_size, batch_first=True
        )
        self.market_encoder = nn.GRU(
            spec.n_market_features, asset_hidden_size, batch_first=True
        )
        if architecture == "attention":
            self.asset_temporal_attention = nn.Linear(asset_hidden_size, 1)
            self.market_temporal_attention = nn.Linear(asset_hidden_size, 1)
            self.market_projection = _layer_init(
                nn.Linear(asset_hidden_size, asset_hidden_size), gain=1.0
            )
            self.portfolio_projection = _layer_init(
                nn.Linear(
                    spec.per_asset_portfolio_features, asset_hidden_size
                ),
                gain=1.0,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=asset_hidden_size,
                nhead=attention_heads,
                dim_feedforward=asset_hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.cross_asset_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=attention_layers, enable_nested_tensor=False
            )
            self.cross_asset_norm = nn.LayerNorm(asset_hidden_size)
        per_asset_input = (
            asset_hidden_size * 3 + spec.per_asset_portfolio_features
        )
        self.stock_actor = nn.Sequential(
            _layer_init(nn.Linear(per_asset_input, context_hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(context_hidden_size, 1), gain=0.01),
        )
        global_input = (
            asset_hidden_size * 2 + spec.global_portfolio_features
        )
        self.cash_actor = nn.Sequential(
            _layer_init(nn.Linear(global_input, context_hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(context_hidden_size, 1), gain=0.01),
        )
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(global_input, context_hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(context_hidden_size, context_hidden_size // 2)),
            nn.Tanh(),
            _layer_init(nn.Linear(context_hidden_size // 2, 1), gain=1.0),
        )
        # Shared exploration scales keep the policy permutation-equivariant and
        # let it generalize to constituents that enter VN30 after training.
        self.stock_log_std = nn.Parameter(torch.tensor(-0.5))
        self.cash_log_std = nn.Parameter(torch.tensor(-0.5))

    def _parse(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        batch = observations.shape[0]
        offset = 0
        asset_end = offset + self.spec.asset_history_size
        asset_history = observations[:, offset:asset_end].reshape(
            batch,
            self.spec.n_assets,
            self.spec.lookback,
            self.spec.n_stock_features,
        )
        offset = asset_end
        market_end = offset + self.spec.market_history_size
        market_history = observations[:, offset:market_end].reshape(
            batch, self.spec.lookback, self.spec.n_market_features
        )
        offset = market_end
        portfolio_size = (
            self.spec.n_assets * self.spec.per_asset_portfolio_features
        )
        portfolio = observations[:, offset : offset + portfolio_size].reshape(
            batch,
            self.spec.n_assets,
            self.spec.per_asset_portfolio_features,
        )
        offset += portfolio_size
        globals_ = observations[:, offset:]
        return asset_history, market_history, portfolio, globals_

    def _features(
        self, observations: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        asset_history, market_history, portfolio, globals_ = self._parse(observations)
        batch = asset_history.shape[0]
        normalized_assets = self.asset_norm(asset_history).reshape(
            batch * self.spec.n_assets,
            self.spec.lookback,
            self.spec.n_stock_features,
        )
        asset_sequence, asset_last = self.asset_encoder(normalized_assets)
        market_sequence, market_last = self.market_encoder(
            self.market_norm(market_history)
        )
        if self.architecture == "attention":
            asset_scores = self.asset_temporal_attention(asset_sequence).squeeze(-1)
            asset_weights = torch.softmax(asset_scores, dim=1).unsqueeze(-1)
            asset_hidden = (asset_sequence * asset_weights).sum(dim=1).reshape(
                batch, self.spec.n_assets, self.asset_hidden_size
            )
            market_scores = self.market_temporal_attention(
                market_sequence
            ).squeeze(-1)
            market_weights = torch.softmax(market_scores, dim=1).unsqueeze(-1)
            market_hidden = (market_sequence * market_weights).sum(dim=1)
        else:
            asset_hidden = asset_last[-1].reshape(
                batch, self.spec.n_assets, self.asset_hidden_size
            )
            market_hidden = market_last[-1]
        tradable = portfolio[:, :, 0] > 0.5
        if self.architecture == "attention":
            market_context = self.market_projection(market_hidden)[:, None, :]
            portfolio_context = self.portfolio_projection(portfolio)
            cross_input = (
                asset_hidden
                + market_context
                + portfolio_context
            )
            attention_padding_mask = ~tradable
            all_assets_masked = attention_padding_mask.all(dim=1)
            if all_assets_masked.any():
                attention_padding_mask = attention_padding_mask.clone()
                attention_padding_mask[all_assets_masked, 0] = False
            asset_hidden = self.cross_asset_encoder(
                cross_input, src_key_padding_mask=attention_padding_mask
            )
            asset_hidden = self.cross_asset_norm(asset_hidden)
        mask_float = tradable.to(asset_hidden.dtype).unsqueeze(-1)
        pooled_assets = (asset_hidden * mask_float).sum(dim=1) / mask_float.sum(
            dim=1
        ).clamp_min(1.0)
        return asset_hidden, market_hidden, pooled_assets, portfolio, globals_, tradable

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            asset_hidden,
            market_hidden,
            pooled_assets,
            portfolio,
            globals_,
            tradable,
        ) = self._features(observations)
        n_assets = self.spec.n_assets
        repeated_market = market_hidden[:, None, :].expand(-1, n_assets, -1)
        repeated_pool = pooled_assets[:, None, :].expand(-1, n_assets, -1)
        stock_input = torch.cat(
            [asset_hidden, repeated_market, repeated_pool, portfolio], dim=-1
        )
        stock_means = self.stock_actor(stock_input).squeeze(-1)
        global_input = torch.cat([market_hidden, pooled_assets, globals_], dim=-1)
        cash_mean = self.cash_actor(global_input)
        if self.momentum_prior_topk > 0:
            asset_history, _, _, _ = self._parse(observations)
            # return_60d is feature index 3 in STOCK_FEATURE_NAMES.  Only the
            # latest fully observed day is used, so the prior has no look-ahead.
            momentum = asset_history[:, :, -1, 3]
            masked_momentum = torch.where(
                tradable,
                momentum,
                torch.full_like(momentum, -torch.inf),
            )
            topk = min(self.momentum_prior_topk, self.spec.n_assets)
            selected_indices = torch.topk(
                masked_momentum, k=topk, dim=1
            ).indices
            selected = torch.zeros_like(tradable)
            selected.scatter_(1, selected_indices, True)
            selected &= tradable
            selected_count = selected.sum(dim=1, keepdim=True)
            selected_weight = (
                (1.0 - self.momentum_prior_cash_weight)
                / selected_count.clamp_min(1).to(stock_means.dtype)
            )
            prior_stock_logits = torch.full_like(stock_means, -10.0)
            prior_stock_logits = torch.where(
                selected,
                selected_weight.log().expand_as(stock_means),
                prior_stock_logits,
            )
            prior_cash_logits = torch.full_like(
                cash_mean, math.log(self.momentum_prior_cash_weight)
            )
            prior_cash_logits = torch.where(
                selected_count > 0,
                prior_cash_logits,
                torch.zeros_like(prior_cash_logits),
            )
            stock_means = (
                stock_means + self.momentum_prior_strength * prior_stock_logits
            )
            cash_mean = (
                cash_mean + self.momentum_prior_strength * prior_cash_logits
            )
        stock_means = torch.where(
            tradable, stock_means, torch.full_like(stock_means, -10.0)
        )
        means = torch.cat([stock_means, cash_mean], dim=-1).clamp(-10.0, 10.0)
        values = self.critic(global_input).squeeze(-1)
        action_mask = torch.cat(
            [
                tradable,
                torch.ones(
                    (tradable.shape[0], 1),
                    dtype=torch.bool,
                    device=tradable.device,
                ),
            ],
            dim=-1,
        )
        return means, values, action_mask

    def distribution(
        self, observations: torch.Tensor
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        means, values, action_mask = self(observations)
        stock_log_std = self.stock_log_std.expand(self.spec.n_assets)
        log_std = torch.cat([stock_log_std, self.cash_log_std.reshape(1)])
        std = log_std.clamp(-4.0, 1.0).exp().expand_as(means)
        return Normal(means, std), values, action_mask

    @staticmethod
    def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask.to(values.dtype)).sum(dim=-1)

    @torch.no_grad()
    def act(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values, action_mask = self.distribution(observations)
        actions = distribution.mean if deterministic else distribution.sample()
        log_prob = self._masked_sum(distribution.log_prob(actions), action_mask)
        return actions, log_prob, values

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values, action_mask = self.distribution(observations)
        log_prob = self._masked_sum(distribution.log_prob(actions), action_mask)
        entropy = self._masked_sum(distribution.entropy(), action_mask)
        if self.normalize_entropy_by_action_dim:
            entropy = entropy / action_mask.sum(dim=-1).clamp_min(1)
        return log_prob, entropy, values


@dataclass(slots=True)
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    transition_discounts: np.ndarray
    trace_discounts: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class PPOAgent:
    def __init__(self, spec: ObservationSpec, config: PPOConfig | None = None) -> None:
        self.spec = spec
        self.config = config or PPOConfig()
        self.config.validate()
        self.device = self._resolve_device(self.config.device)
        configure_training_device(self.device)
        self._set_seed(self.config.seed)
        self.model = PortfolioActorCritic(
            spec,
            asset_hidden_size=self.config.asset_hidden_size,
            context_hidden_size=self.config.context_hidden_size,
            architecture=self.config.architecture,
            attention_heads=self.config.attention_heads,
            attention_layers=self.config.attention_layers,
            dropout=self.config.dropout,
            momentum_prior_topk=self.config.momentum_prior_topk,
            momentum_prior_cash_weight=self.config.momentum_prior_cash_weight,
            momentum_prior_strength=self.config.momentum_prior_strength,
            normalize_entropy_by_action_dim=(
                self.config.normalize_entropy_by_action_dim
            ),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        return resolve_training_device(requested)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        actions, _, _ = self.model.act(tensor, deterministic=deterministic)
        result = actions.squeeze(0).cpu().numpy()
        if was_training:
            self.model.train()
        return result

    def _value(self, observation: np.ndarray) -> float:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, values, _ = self.model(tensor)
        return float(values.item())

    def _step_decision(
        self, env: VietnamStockEnv, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute one policy decision and hold it until the next review date."""

        total_reward = 0.0
        daily_discount = 1.0
        next_observation: np.ndarray | None = None
        done = False
        info: dict[str, Any] = {}
        market_days = 0
        for offset in range(self.config.rebalance_interval):
            # ``None`` is an explicit exchange-level HOLD. Reconstructing the
            # previous close weights and executing them at today's open would
            # accidentally rebalance whenever there is an overnight gap.
            executed_action = action if offset == 0 else None
            next_observation, reward, done, info = env.step(executed_action)
            total_reward += daily_discount * reward
            daily_discount *= self.config.gamma
            market_days += 1
            if done:
                break
        if next_observation is None:
            raise RuntimeError("A portfolio decision executed zero market days")
        info = dict(info)
        info["decision_market_days"] = market_days
        info["decision_discounted_reward"] = total_reward
        info["decision_transition_discount"] = self.config.gamma**market_days
        info["decision_trace_discount"] = (
            self.config.gamma * self.config.gae_lambda
        ) ** market_days
        return next_observation, total_reward, done, info

    def _collect_rollout(
        self,
        env: VietnamStockEnv,
        observation: np.ndarray,
        steps: int,
    ) -> tuple[RolloutBatch, np.ndarray, list[dict[str, Any]]]:
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        log_probs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        transition_discounts: list[float] = []
        trace_discounts: list[float] = []
        episode_summaries: list[dict[str, Any]] = []
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(steps):
            tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = self.model.act(tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
            next_observation, reward, done, info = self._step_decision(
                env, action
            )
            observations.append(observation.copy())
            actions.append(action)
            log_probs.append(float(log_prob_tensor.item()))
            values.append(float(value_tensor.item()))
            rewards.append(float(reward))
            dones.append(float(done))
            transition_discounts.append(
                float(info["decision_transition_discount"])
            )
            trace_discounts.append(float(info["decision_trace_discount"]))
            episode_reward += reward
            episode_steps += 1
            observation = next_observation
            if done:
                episode_summaries.append(
                    {
                        "episode_reward": episode_reward,
                        "episode_steps": episode_steps,
                        "episode_final_nav": info["nav"],
                    }
                )
                observation = env.reset()
                episode_reward = 0.0
                episode_steps = 0

        last_value = self._value(observation)
        reward_array = np.asarray(rewards, dtype=np.float32)
        value_array = np.asarray(values, dtype=np.float32)
        done_array = np.asarray(dones, dtype=np.float32)
        transition_discount_array = np.asarray(
            transition_discounts, dtype=np.float32
        )
        trace_discount_array = np.asarray(
            trace_discounts, dtype=np.float32
        )
        advantages = np.zeros_like(reward_array)
        last_gae = 0.0
        for index in reversed(range(steps)):
            if index == steps - 1:
                next_value = last_value
            else:
                next_value = value_array[index + 1]
            non_terminal = 1.0 - done_array[index]
            delta = (
                reward_array[index]
                + transition_discount_array[index]
                * next_value
                * non_terminal
                - value_array[index]
            )
            last_gae = (
                delta
                + trace_discount_array[index]
                * non_terminal
                * last_gae
            )
            advantages[index] = last_gae
        returns = advantages + value_array
        batch = RolloutBatch(
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=reward_array,
            old_log_probs=np.asarray(log_probs, dtype=np.float32),
            old_values=value_array,
            transition_discounts=transition_discount_array,
            trace_discounts=trace_discount_array,
            advantages=advantages,
            returns=returns,
        )
        return batch, observation, episode_summaries

    def _update(
        self, batch: RolloutBatch, *, entropy_coefficient: float
    ) -> dict[str, float]:
        n_samples = len(batch.observations)
        advantages = batch.advantages.copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        indices = np.arange(n_samples)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        stopped_early = False

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.config.batch_size):
                mb = indices[start : start + self.config.batch_size]
                obs = torch.as_tensor(
                    batch.observations[mb], dtype=torch.float32, device=self.device
                )
                actions = torch.as_tensor(
                    batch.actions[mb], dtype=torch.float32, device=self.device
                )
                old_log_probs = torch.as_tensor(
                    batch.old_log_probs[mb], dtype=torch.float32, device=self.device
                )
                old_values = torch.as_tensor(
                    batch.old_values[mb], dtype=torch.float32, device=self.device
                )
                returns = torch.as_tensor(
                    batch.returns[mb], dtype=torch.float32, device=self.device
                )
                mb_advantages = torch.as_tensor(
                    advantages[mb], dtype=torch.float32, device=self.device
                )

                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    obs, actions
                )
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                unclipped = -mb_advantages * ratio
                clipped = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_range,
                    1.0 + self.config.clip_range,
                )
                policy_loss = torch.maximum(unclipped, clipped).mean()

                clipped_values = old_values + torch.clamp(
                    new_values - old_values,
                    -self.config.value_clip_range,
                    self.config.value_clip_range,
                )
                value_loss_unclipped = (new_values - returns).pow(2)
                value_loss_clipped = (clipped_values - returns).pow(2)
                value_loss = 0.5 * torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                ).mean()
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - entropy_coefficient * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_range)
                        .float()
                        .mean()
                    )
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_losses.append(float(entropy_mean.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))

            if approx_kls and approx_kls[-1] > self.config.target_kl:
                stopped_early = True
                break

        return_variance = float(np.var(batch.returns))
        explained_variance = (
            1.0
            - float(np.var(batch.returns - batch.old_values)) / return_variance
            if return_variance > 1e-12
            else float("nan")
        )
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "early_stop_kl": float(stopped_early),
            "explained_variance": explained_variance,
        }

    def train(
        self,
        env: VietnamStockEnv,
        *,
        validation_env: VietnamStockEnv | None = None,
        evaluation_interval: int = 10,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Train PPO and restore the best multi-objective validation checkpoint."""

        if env.spec != self.spec:
            raise ValueError("Environment observation specification does not match model")
        if evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive")
        observation = env.reset(seed=self.config.seed)
        timesteps = 0
        update_index = 0
        history: list[dict[str, float]] = []
        best_validation = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        training_started = time.perf_counter()
        target_turnover_penalty = env.config.turnover_penalty_rate
        target_drawdown_penalty = env.config.drawdown_penalty

        while timesteps < self.config.total_timesteps:
            update_started = time.perf_counter()
            rollout_size = min(
                self.config.rollout_steps, self.config.total_timesteps - timesteps
            )
            if self.config.penalty_warmup_fraction > 0:
                curriculum_progress = min(
                    1.0,
                    (timesteps + rollout_size)
                    / (
                        self.config.total_timesteps
                        * self.config.penalty_warmup_fraction
                    ),
                )
            else:
                curriculum_progress = 1.0
            env.config.turnover_penalty_rate = (
                target_turnover_penalty * curriculum_progress
            )
            env.config.drawdown_penalty = (
                target_drawdown_penalty * curriculum_progress
            )
            batch, observation, episodes = self._collect_rollout(
                env, observation, rollout_size
            )
            scheduled_progress = min(
                1.0,
                (timesteps + rollout_size) / self.config.total_timesteps,
            )
            learning_rate_multiplier = 1.0 - scheduled_progress * (
                1.0 - self.config.final_learning_rate_fraction
            )
            learning_rate = self.config.learning_rate * learning_rate_multiplier
            entropy_coefficient = self.config.entropy_coef + scheduled_progress * (
                self.config.final_entropy_coef - self.config.entropy_coef
            )
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = learning_rate
            losses = self._update(
                batch, entropy_coefficient=entropy_coefficient
            )
            timesteps += rollout_size
            update_index += 1
            row = {
                "timesteps": float(timesteps),
                "update": float(update_index),
                "mean_reward": float(np.mean(batch.rewards)),
                "learning_rate": learning_rate,
                "entropy_coefficient": entropy_coefficient,
                "reward_penalty_curriculum": curriculum_progress,
                **losses,
            }
            if episodes:
                row["episode_final_nav"] = float(
                    np.mean([item["episode_final_nav"] for item in episodes])
                )

            should_evaluate = validation_env is not None and (
                update_index % evaluation_interval == 0
                or timesteps >= self.config.total_timesteps
            )
            if should_evaluate:
                from .evaluation import evaluate_agent

                _, validation_metrics = evaluate_agent(validation_env, self)
                validation_sharpe = float(validation_metrics["sharpe_ratio"])
                validation_cagr = float(validation_metrics["cagr"])
                validation_turnover = float(
                    validation_metrics["annual_turnover"]
                )
                validation_drawdown = abs(
                    float(validation_metrics["maximum_drawdown"])
                )
                validation_score = _validation_checkpoint_score(
                    validation_metrics, self.config
                )
                row.update(
                    {
                        "validation_sharpe": validation_sharpe,
                        "validation_cagr": validation_cagr,
                        "validation_turnover": validation_turnover,
                        "validation_drawdown": validation_drawdown,
                        "validation_score": validation_score,
                    }
                )
                if validation_score > best_validation:
                    best_validation = validation_score
                    best_state = copy.deepcopy(self.model.state_dict())
            update_seconds = time.perf_counter() - update_started
            elapsed_seconds = time.perf_counter() - training_started
            progress = timesteps / self.config.total_timesteps
            steps_per_second = timesteps / max(elapsed_seconds, 1e-12)
            eta_seconds = (
                (self.config.total_timesteps - timesteps) / steps_per_second
                if steps_per_second > 0
                else float("nan")
            )
            row.update(
                {
                    "progress_percent": progress * 100.0,
                    "update_seconds": update_seconds,
                    "elapsed_seconds": elapsed_seconds,
                    "eta_seconds": eta_seconds,
                    "steps_per_second": steps_per_second,
                }
            )
            history.append(row)
            if verbose:
                validation_text = (
                    f", val_score={row['validation_score']:.3f} "
                    f"val_sharpe={row['validation_sharpe']:.3f} "
                    f"val_turnover={row['validation_turnover']:.2f}"
                    if "validation_sharpe" in row
                    else ""
                )
                print(
                    f"update={update_index:04d} "
                    f"progress={progress:6.2%} "
                    f"steps={timesteps:7d}/{self.config.total_timesteps} "
                    f"elapsed={format_duration(elapsed_seconds)} "
                    f"eta={format_duration(eta_seconds)} "
                    f"update_time={update_seconds:6.1f}s "
                    f"lr={learning_rate:.2e} "
                    f"policy={losses['policy_loss']:.4f} "
                    f"value={losses['value_loss']:.4f}{validation_text}",
                    flush=True,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif validation_env is not None:
            raise RuntimeError(
                "No checkpoint produced finite validation metrics; refusing "
                "to save an unselected final policy"
            )
        env.config.turnover_penalty_rate = target_turnover_penalty
        env.config.drawdown_penalty = target_drawdown_penalty
        if verbose:
            total_seconds = time.perf_counter() - training_started
            print(
                f"Training seed {self.config.seed} completed in "
                f"{format_duration(total_seconds)} "
                f"({total_seconds:.1f}s).",
                flush=True,
            )
        return history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_version": MODEL_VERSION,
                "training_algorithm_version": TRAINING_ALGORITHM_VERSION,
                "state_dict": self.model.state_dict(),
                "ppo_config": dataclass_dict(self.config),
                "observation_spec": {
                    "n_assets": self.spec.n_assets,
                    "lookback": self.spec.lookback,
                    "n_stock_features": self.spec.n_stock_features,
                    "n_market_features": self.spec.n_market_features,
                    "settlement_days": self.spec.settlement_days,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "PPOAgent":
        resolved = cls._resolve_device(device)
        checkpoint = torch.load(path, map_location=resolved, weights_only=False)
        checkpoint_model_version = checkpoint.get("model_version", 1)
        checkpoint_algorithm_version = checkpoint.get(
            "training_algorithm_version", 1
        )
        if (
            checkpoint_model_version != MODEL_VERSION
            or checkpoint_algorithm_version != TRAINING_ALGORITHM_VERSION
        ):
            raise ValueError(
                "Checkpoint version mismatch: "
                f"model={checkpoint_model_version}, "
                f"algorithm={checkpoint_algorithm_version}; expected "
                f"model={MODEL_VERSION}, algorithm={TRAINING_ALGORITHM_VERSION}. "
                "Retrain with the current code."
            )
        spec = ObservationSpec(**checkpoint["observation_spec"])
        config_data = dict(checkpoint["ppo_config"])
        # Version-2 checkpoints used the full hard-coded prior.
        config_data.setdefault("momentum_prior_strength", 1.0)
        config_data["device"] = device
        config = PPOConfig(**config_data)
        agent = cls(spec, config)
        agent.model.load_state_dict(checkpoint["state_dict"])
        agent.model.eval()
        return agent
