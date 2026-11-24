from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import MarketConfig
from .data import MarketDataBundle


@dataclass(slots=True, frozen=True)
class ObservationSpec:
    n_assets: int
    lookback: int
    n_stock_features: int
    n_market_features: int
    settlement_days: int

    @property
    def asset_history_size(self) -> int:
        return self.n_assets * self.lookback * self.n_stock_features

    @property
    def market_history_size(self) -> int:
        return self.lookback * self.n_market_features

    @property
    def per_asset_portfolio_features(self) -> int:
        # Tradable flag, available holdings, then each pending-buy bucket.
        return 2 + self.settlement_days

    @property
    def global_portfolio_features(self) -> int:
        # Cash, pending-sale buckets, drawdown, NAV growth, last turnover/cost.
        return self.settlement_days + 5

    @property
    def observation_dim(self) -> int:
        portfolio = self.n_assets * self.per_asset_portfolio_features
        return (
            self.asset_history_size
            + self.market_history_size
            + portfolio
            + self.global_portfolio_features
        )

    @property
    def action_dim(self) -> int:
        # One allocation logit per stock and one for cash.
        return self.n_assets + 1


@dataclass(slots=True)
class PendingBuy:
    due_index: int
    shares: np.ndarray


@dataclass(slots=True)
class PendingSale:
    due_index: int
    cash: float


class VietnamStockEnv:
    """Daily VN30 portfolio simulator with explicit T+2 settlement ledgers.

    Timeline for date T:

    1. trades due on T are settled before the agent acts;
    2. the observation contains market data only through T-1;
    3. target weights are executed using T's open plus configurable slippage;
    4. the portfolio is marked to T's close and a net-return reward is emitted.

    A purchase removes cash immediately but its shares cannot be sold until
    T+2. A sale removes shares immediately but its net proceeds cannot fund a
    purchase until T+2. Pending shares remain exposed to market price changes.
    """

    def __init__(
        self,
        data: MarketDataBundle,
        *,
        start_date: str,
        end_date: str,
        lookback: int,
        config: MarketConfig | None = None,
        random_start: bool = False,
        episode_days: int | None = None,
        initial_cash_delay_days: int = 0,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.config = config or MarketConfig()
        self.config.validate()
        self.lookback = int(lookback)
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        period = data.period_indices(start_date, end_date)
        period = period[period >= self.lookback]
        if len(period) < 2:
            raise ValueError("The selected period is too short for the lookback")
        self.period_start = int(period[0])
        self.period_end = int(period[-1])
        self.random_start = bool(random_start)
        self.episode_days = episode_days
        self.initial_cash_delay_days = int(initial_cash_delay_days)
        if self.initial_cash_delay_days < 0:
            raise ValueError("initial_cash_delay_days cannot be negative")
        self.rng = np.random.default_rng(seed)
        self.spec = ObservationSpec(
            n_assets=data.n_assets,
            lookback=self.lookback,
            n_stock_features=data.stock_features.shape[-1],
            n_market_features=data.market_features.shape[-1],
            settlement_days=self.config.settlement_days,
        )

        self.current_index = self.period_start
        self.episode_end = self.period_end
        self.available_cash = self.config.initial_cash
        self.available_shares = np.zeros(data.n_assets, dtype=np.float64)
        self.pending_buys: list[PendingBuy] = []
        self.pending_sales: list[PendingSale] = []
        self.last_nav = self.config.initial_cash
        self.peak_nav = self.config.initial_cash
        self.total_fees = 0.0
        self.total_tax = 0.0
        self.total_slippage = 0.0
        self.total_trade_count = 0
        self.episode_step = 0
        self.last_turnover = 0.0
        self.last_transaction_cost_ratio = 0.0

    @property
    def observation_dim(self) -> int:
        return self.spec.observation_dim

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def current_date(self):
        return self.data.dates[self.current_index]

    @property
    def current_tradable_mask(self) -> np.ndarray:
        return self.data.tradable_mask[self.current_index].copy()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.random_start and self.episode_days:
            latest = max(self.period_start, self.period_end - self.episode_days + 1)
            self.current_index = int(self.rng.integers(self.period_start, latest + 1))
            self.episode_end = min(
                self.current_index + self.episode_days - 1, self.period_end
            )
        else:
            self.current_index = self.period_start
            self.episode_end = self.period_end

        self.available_cash = (
            0.0
            if self.initial_cash_delay_days > 0
            else float(self.config.initial_cash)
        )
        self.available_shares = np.zeros(self.data.n_assets, dtype=np.float64)
        self.pending_buys = []
        self.pending_sales = (
            [
                PendingSale(
                    due_index=self.current_index + self.initial_cash_delay_days,
                    cash=float(self.config.initial_cash),
                )
            ]
            if self.initial_cash_delay_days > 0
            else []
        )
        self.last_nav = float(self.config.initial_cash)
        self.peak_nav = float(self.config.initial_cash)
        self.total_fees = 0.0
        self.total_tax = 0.0
        self.total_slippage = 0.0
        self.total_trade_count = 0
        self.episode_step = 0
        self.last_turnover = 0.0
        self.last_transaction_cost_ratio = 0.0
        return self._observation()

    def _pending_share_total(self) -> np.ndarray:
        if not self.pending_buys:
            return np.zeros(self.data.n_assets, dtype=np.float64)
        return np.sum([item.shares for item in self.pending_buys], axis=0)

    def _pending_cash_total(self) -> float:
        return float(sum(item.cash for item in self.pending_sales))

    def _settle(self, index: int) -> None:
        remaining_buys: list[PendingBuy] = []
        for item in self.pending_buys:
            if item.due_index <= index:
                self.available_shares += item.shares
            else:
                remaining_buys.append(item)
        self.pending_buys = remaining_buys

        remaining_sales: list[PendingSale] = []
        for item in self.pending_sales:
            if item.due_index <= index:
                self.available_cash += item.cash
            else:
                remaining_sales.append(item)
        self.pending_sales = remaining_sales

    def _mark_prices(self, index: int) -> np.ndarray:
        return self.data.valuation_prices[index] * self.config.price_multiplier

    def _lot_size(self, index: int) -> int:
        effective = np.datetime64(self.config.lot_size_100_effective_date)
        return (
            self.config.lot_size
            if self.data.dates[index].to_datetime64() >= effective
            else self.config.legacy_lot_size
        )

    def _odd_lot_trading_available(self, index: int) -> bool:
        effective = np.datetime64(
            self.config.odd_lot_trading_effective_date
        )
        return self.data.dates[index].to_datetime64() >= effective

    def _portfolio_nav(self, index: int) -> float:
        shares = self.available_shares + self._pending_share_total()
        stock_value = float(np.dot(shares, self._mark_prices(index)))
        return self.available_cash + self._pending_cash_total() + stock_value

    def _portfolio_state(self) -> tuple[np.ndarray, np.ndarray]:
        mark_index = max(self.current_index - 1, 0)
        prices = self._mark_prices(mark_index)
        nav = max(self._portfolio_nav(mark_index), 1e-12)
        available_weights = self.available_shares * prices / nav
        pending_weight_buckets = np.zeros(
            (self.config.settlement_days, self.data.n_assets), dtype=np.float64
        )
        for item in self.pending_buys:
            remaining = max(1, item.due_index - self.current_index)
            bucket = min(remaining, self.config.settlement_days) - 1
            if self.config.settlement_days:
                pending_weight_buckets[bucket] += item.shares * prices / nav

        pending_cash_buckets = np.zeros(self.config.settlement_days, dtype=np.float64)
        for item in self.pending_sales:
            remaining = max(1, item.due_index - self.current_index)
            bucket = min(remaining, self.config.settlement_days) - 1
            if self.config.settlement_days:
                pending_cash_buckets[bucket] += item.cash / nav

        drawdown = max(0.0, 1.0 - self.last_nav / max(self.peak_nav, 1e-12))
        global_features = np.concatenate(
            [
                np.asarray([self.available_cash / nav]),
                pending_cash_buckets,
                np.asarray(
                    [
                        drawdown,
                        np.log(max(nav, 1e-12) / self.config.initial_cash),
                        self.last_turnover,
                        self.last_transaction_cost_ratio,
                    ]
                ),
            ]
        )
        per_asset = np.column_stack(
            [
                self.current_tradable_mask.astype(np.float64),
                available_weights,
                pending_weight_buckets.T,
            ]
        )
        return per_asset, global_features

    def _observation(self) -> np.ndarray:
        start = self.current_index - self.lookback
        stop = self.current_index
        asset_history = self.data.stock_features[start:stop]
        # Convert [time, asset, feature] to [asset, time, feature].
        asset_history = np.transpose(asset_history, (1, 0, 2))
        market_history = self.data.market_features[start:stop]
        per_asset, global_features = self._portfolio_state()
        observation = np.concatenate(
            [
                asset_history.reshape(-1),
                market_history.reshape(-1),
                per_asset.reshape(-1),
                global_features,
            ]
        ).astype(np.float32)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                f"Observation shape {observation.shape} != {(self.observation_dim,)}"
            )
        return observation

    def action_to_weights(self, action: np.ndarray) -> np.ndarray:
        logits = np.asarray(action, dtype=np.float64).reshape(-1)
        if logits.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape {(self.action_dim,)}, got {logits.shape}")
        logits = np.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = np.clip(logits, -20.0, 20.0)
        logits[:-1][~self.current_tradable_mask] = -1e9
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        weights = exp_logits / max(exp_logits.sum(), 1e-12)
        return weights

    @staticmethod
    def weights_to_action(weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.clip(weights, 1e-12, None)
        weights = weights / weights.sum()
        return np.log(weights)

    def current_hold_weights(self) -> np.ndarray:
        """Return the observable T-1 marked portfolio without placing an order."""

        # The ensemble must not use today's opening prices while constructing
        # a decision. Holdings are valued at the last completed close instead.
        prices = self._mark_prices(max(self.current_index - 1, 0))
        shares = self.available_shares + self._pending_share_total()
        stock_values = shares * prices
        cash_value = self.available_cash + self._pending_cash_total()
        values = np.concatenate([stock_values, [cash_value]])
        if values.sum() <= 0:
            values[-1] = 1.0
        return values / values.sum()

    def current_hold_weights_action(self) -> np.ndarray:
        """Compatibility representation of the observable current portfolio."""

        return self.weights_to_action(self.current_hold_weights())

    def _execution_prices(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        open_prices = self.data.open_prices[index] * self.config.price_multiplier
        previous = self._mark_prices(max(index - 1, 0))
        slippage = self.config.slippage_bps / 10_000.0
        lower_limit = previous * (1.0 - self.config.price_limit_rate)
        upper_limit = previous * (1.0 + self.config.price_limit_rate)
        # Only information observable at the open is used.  Expanding the
        # simple limit band to contain the observed open handles adjusted-price
        # corporate actions without consulting today's high or low.
        effective_low = np.minimum(lower_limit, open_prices)
        effective_high = np.maximum(upper_limit, open_prices)
        buy = np.minimum(open_prices * (1.0 + slippage), effective_high)
        sell = np.maximum(open_prices * (1.0 - slippage), effective_low)
        mask = self.data.execution_mask[index]
        buy = np.where(mask & np.isfinite(buy), buy, 0.0)
        sell = np.where(mask & np.isfinite(sell), sell, 0.0)
        return buy, sell

    def _execute(
        self,
        target_weights: np.ndarray,
        *,
        hold_positions: bool = False,
    ) -> dict[str, Any]:
        index = self.current_index
        decision_mask = self.current_tradable_mask
        execution_mask = self.data.execution_mask[index]
        buy_prices, sell_prices = self._execution_prices(index)
        lot_size = self._lot_size(index)
        open_mark = self.data.open_prices[index] * self.config.price_multiplier
        previous_mark = self._mark_prices(max(index - 1, 0))
        # A suspended/missing opening print does not make an existing position
        # worthless. Mark it at the latest completed close until it can trade.
        pretrade_mark = np.where(
            execution_mask & np.isfinite(open_mark) & (open_mark > 0),
            open_mark,
            previous_mark,
        )
        owned_shares = self.available_shares + self._pending_share_total()
        pretrade_nav = (
            self.available_cash
            + self._pending_cash_total()
            + float(np.dot(owned_shares, np.nan_to_num(pretrade_mark, nan=0.0)))
        )
        buy_mask = decision_mask & execution_mask
        if hold_positions:
            # HOLD means no target-weight rebalancing at the current open.  In
            # particular, an overnight price gap must not create a synthetic
            # trade on a day between policy reviews. Former constituents remain
            # a forced exit under the point-in-time VN30 mandate.
            desired_shares = owned_shares.copy()
            desired_shares[~self.data.universe_mask[index]] = 0.0
        else:
            safe_prices = np.where(
                buy_mask & (open_mark > 0), open_mark, np.inf
            )
            desired_lots = (
                (target_weights[:-1] * pretrade_nav / safe_prices)
                / lot_size
            )
            desired_shares = np.floor(desired_lots + 1e-5) * lot_size
            desired_shares = np.nan_to_num(
                desired_shares, nan=0.0, posinf=0.0
            )
            # If yesterday did not provide enough public information for a
            # decision, freeze an existing current-constituent position. This
            # avoids interpreting an unavailable target as an instruction to
            # liquidate it. A former constituent is deliberately not frozen.
            frozen = self.data.universe_mask[index] & ~decision_mask
            desired_shares[frozen] = owned_shares[frozen]
        delta = desired_shares - owned_shares
        requested_trade_fraction = (
            np.abs(delta) * np.nan_to_num(pretrade_mark, nan=0.0)
            / max(pretrade_nav, 1e-12)
        )
        delta = np.where(
            requested_trade_fraction >= self.config.minimum_trade_fraction,
            delta,
            0.0,
        )

        # Same-day total volume is unknown at the open. Use the most recent
        # completed session as the ex-ante liquidity budget.
        volume_reference = self.data.volumes[max(index - 1, 0)]
        volume_cap = np.floor(
            (volume_reference * self.config.max_volume_fraction)
            / lot_size
        ) * lot_size
        sell_volume_cap = np.where(execution_mask, volume_cap, 0.0)
        buy_volume_cap = np.where(buy_mask, volume_cap, 0.0)

        sell_shares = np.minimum(np.maximum(-delta, 0.0), self.available_shares)
        # Former constituents may always be liquidated when an opening price is
        # available, even though they are no longer eligible for a new purchase.
        sell_shares = np.minimum(sell_shares, sell_volume_cap)
        if self._odd_lot_trading_available(index):
            # HOSE's odd-lot board can execute the residual 1..99 shares. The
            # simulator uses the same conservative open/slippage assumption
            # for the regular and odd portions of a sell order.
            sell_shares = np.floor(sell_shares + 1e-8)
        else:
            sell_shares = np.floor(sell_shares / lot_size) * lot_size
        sell_notional_by_asset = sell_shares * sell_prices
        sell_notional = float(sell_notional_by_asset.sum())
        sell_fee = sell_notional * self.config.broker_fee_rate
        sell_tax = sell_notional * self.config.sell_tax_rate
        sell_receivable = sell_notional - sell_fee - sell_tax
        if sell_notional > 0:
            self.available_shares -= sell_shares
            self.pending_sales.append(
                PendingSale(
                    due_index=index + self.config.settlement_days,
                    cash=sell_receivable,
                )
            )

        buy_shares = np.minimum(np.maximum(delta, 0.0), buy_volume_cap)
        buy_shares = np.floor(buy_shares / lot_size) * lot_size
        unit_cost = buy_prices * (1.0 + self.config.broker_fee_rate)
        requested_cost = float(np.dot(buy_shares, unit_cost))
        if requested_cost > self.available_cash and requested_cost > 0:
            scale = self.available_cash / requested_cost
            buy_shares = (
                np.floor((buy_shares * scale) / lot_size)
                * lot_size
            )
        buy_notional_by_asset = buy_shares * buy_prices
        buy_notional = float(buy_notional_by_asset.sum())
        buy_fee = buy_notional * self.config.broker_fee_rate
        buy_slippage_cost = float(
            np.dot(
                buy_shares,
                np.maximum(
                    np.nan_to_num(buy_prices - open_mark, nan=0.0), 0.0
                ),
            )
        )
        sell_slippage_cost = float(
            np.dot(
                sell_shares,
                np.maximum(
                    np.nan_to_num(open_mark - sell_prices, nan=0.0), 0.0
                ),
            )
        )
        slippage_cost = buy_slippage_cost + sell_slippage_cost
        buy_cost = buy_notional + buy_fee
        if buy_cost > self.available_cash + 1e-6:
            raise RuntimeError("Buy allocation exceeded available cash")
        if buy_notional > 0:
            self.available_cash -= buy_cost
            self.pending_buys.append(
                PendingBuy(
                    due_index=index + self.config.settlement_days,
                    shares=buy_shares.copy(),
                )
            )

        if self.config.settlement_days == 0:
            self._settle(index)

        fees = buy_fee + sell_fee
        self.total_fees += fees
        self.total_tax += sell_tax
        self.total_slippage += slippage_cost
        trade_count = int(np.count_nonzero(buy_shares) + np.count_nonzero(sell_shares))
        self.total_trade_count += trade_count
        turnover = (buy_notional + sell_notional) / max(pretrade_nav, 1e-12)
        self.last_turnover = turnover
        self.last_transaction_cost_ratio = (fees + sell_tax + slippage_cost) / max(
            pretrade_nav, 1e-12
        )
        return {
            "pretrade_nav": pretrade_nav,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "broker_fees": fees,
            "sell_tax": sell_tax,
            "slippage_cost": slippage_cost,
            "turnover": turnover,
            "trade_count": trade_count,
            "buy_shares": buy_shares,
            "sell_shares": sell_shares,
        }

    def step(
        self, action: np.ndarray | None
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        hold_positions = action is None
        if hold_positions:
            target_weights = self.current_hold_weights()
            former = ~self.data.universe_mask[self.current_index]
            removed_weight = float(target_weights[:-1][former].sum())
            target_weights[:-1][former] = 0.0
            target_weights[-1] += removed_weight
        else:
            target_weights = self.action_to_weights(action)
        previous_nav = self.last_nav
        previous_peak = self.peak_nav
        execution = self._execute(
            target_weights, hold_positions=hold_positions
        )
        nav = self._portfolio_nav(self.current_index)
        if not np.isfinite(nav) or nav <= 0:
            raise RuntimeError(f"Invalid portfolio NAV: {nav}")
        daily_return = nav / previous_nav - 1.0
        self.peak_nav = max(self.peak_nav, nav)
        drawdown = 1.0 - nav / self.peak_nav
        previous_drawdown = 1.0 - previous_nav / max(previous_peak, 1e-12)
        log_return = np.log(nav / previous_nav)
        turnover_penalty = (
            self.config.turnover_penalty_rate * execution["turnover"]
        )
        drawdown_penalty = self.config.drawdown_penalty * max(
            0.0, drawdown - previous_drawdown
        )
        open_prices = self.data.open_prices[self.current_index]
        close_prices = self.data.valuation_prices[self.current_index]
        valid_returns = (
            self.current_tradable_mask
            & np.isfinite(open_prices)
            & np.isfinite(close_prices)
            & (open_prices > 0)
        )
        intraday_returns = np.zeros(self.data.n_assets, dtype=np.float64)
        intraday_returns[valid_returns] = (
            close_prices[valid_returns] / open_prices[valid_returns] - 1.0
        )
        # Reward the allocation that was actually established, not an order
        # request that settlement, cash, lot or liquidity constraints rejected.
        actual_shares = self.available_shares + self._pending_share_total()
        actual_open_weights = (
            actual_shares
            * np.where(
                np.isfinite(open_prices) & (open_prices > 0),
                open_prices * self.config.price_multiplier,
                0.0,
            )
            / max(execution["pretrade_nav"], 1e-12)
        )
        actual_stock_weights = np.where(
            valid_returns, actual_open_weights, 0.0
        )
        actual_stock_total = float(actual_stock_weights.sum())
        allocation_return_contribution = float(
            np.dot(actual_stock_weights, intraday_returns)
        )
        equal_weight_return = (
            float(intraday_returns[valid_returns].mean())
            if valid_returns.any()
            else 0.0
        )
        # Compare with an equal-weight stock sleeve at the *same realized
        # exposure*. Without this multiplier, a nearly all-cash policy could
        # receive the alpha reward of a fully invested selector and exploit a
        # reward signal disconnected from its actual P&L.
        allocation_alpha = (
            allocation_return_contribution
            - actual_stock_total * equal_weight_return
            if actual_stock_total > 1e-12
            else 0.0
        )
        allocation_alpha_reward = (
            self.config.allocation_alpha_reward * allocation_alpha
        )
        reward = self.config.reward_scale * (
            log_return
            + allocation_alpha_reward
            - turnover_penalty
            - drawdown_penalty
        )
        self.last_nav = nav

        shares = self.available_shares + self._pending_share_total()
        stock_values = shares * self._mark_prices(self.current_index)
        realized_weights = stock_values / nav
        cash_weight = (self.available_cash + self._pending_cash_total()) / nav
        if self.episode_step == 0:
            # Every simulated strategy also begins trading at this open, so the
            # benchmark must not receive an extra overnight return on day one.
            previous_market_close = self.data.market_open[self.current_index]
        else:
            previous_market_close = self.data.market_close[max(self.current_index - 1, 0)]
        market_close = self.data.market_close[self.current_index]
        benchmark_return = (
            market_close / previous_market_close - 1.0
            if previous_market_close > 0 and market_close > 0
            else 0.0
        )

        info: dict[str, Any] = {
            "date": self.current_date,
            "nav": nav,
            "daily_return": daily_return,
            "reward": reward,
            "log_return": log_return,
            "turnover_reward_penalty": turnover_penalty,
            "drawdown_reward_penalty": drawdown_penalty,
            "allocation_alpha": allocation_alpha,
            "allocation_alpha_reward": allocation_alpha_reward,
            "drawdown": drawdown,
            "available_cash": self.available_cash,
            "pending_sale_cash": self._pending_cash_total(),
            "pending_buy_value": float(
                np.dot(self._pending_share_total(), self._mark_prices(self.current_index))
            ),
            "market_open": float(self.data.market_open[self.current_index]),
            "market_close": float(self.data.market_close[self.current_index]),
            "benchmark_return": benchmark_return,
            "target_weights": target_weights.copy(),
            "realized_weights": np.concatenate([realized_weights, [cash_weight]]),
            "available_shares": self.available_shares.copy(),
            "pending_buy_shares": self._pending_share_total(),
            **execution,
        }

        done = self.current_index >= self.episode_end
        self.episode_step += 1
        if not done:
            self.current_index += 1
            self._settle(self.current_index)
            observation = self._observation()
        else:
            observation = self._observation()
        return observation, float(reward), done, info
