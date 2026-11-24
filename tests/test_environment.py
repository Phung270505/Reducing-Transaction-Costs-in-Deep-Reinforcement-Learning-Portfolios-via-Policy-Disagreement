from __future__ import annotations

import numpy as np
import pandas as pd

from vn30ppo.config import MarketConfig
from vn30ppo.data import MarketDataBundle, STOCK_FEATURE_NAMES
from vn30ppo.environment import VietnamStockEnv


def synthetic_bundle(n_assets: int = 2, n_dates: int = 9) -> MarketDataBundle:
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    tickers = tuple(chr(ord("A") + index) for index in range(n_assets))
    prices = np.full((n_dates, n_assets), 10.0, dtype=np.float64)
    volume = np.full((n_dates, n_assets), 1_000_000.0, dtype=np.float64)
    features = np.zeros((n_dates, n_assets, len(STOCK_FEATURE_NAMES)), dtype=np.float32)
    for day in range(n_dates):
        features[day, :, 0] = day
    market_features = np.zeros(
        (n_dates, len(STOCK_FEATURE_NAMES)), dtype=np.float32
    )
    return MarketDataBundle(
        dates=dates,
        tickers=tickers,
        open_prices=prices.copy(),
        high_prices=prices.copy(),
        low_prices=prices.copy(),
        close_prices=prices.copy(),
        valuation_prices=prices.copy(),
        volumes=volume,
        stock_features=features,
        market_features=market_features,
        tradable_mask=np.ones((n_dates, n_assets), dtype=bool),
        execution_mask=np.ones((n_dates, n_assets), dtype=bool),
        universe_mask=np.ones((n_dates, n_assets), dtype=bool),
        market_open=np.full(n_dates, 1_000.0),
        market_close=np.full(n_dates, 1_000.0),
        eligible_assets=np.ones(n_assets, dtype=bool),
    )


def make_env(n_assets: int = 2) -> VietnamStockEnv:
    data = synthetic_bundle(n_assets=n_assets)
    config = MarketConfig(
        initial_cash=100_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=0.0,
        max_volume_fraction=1.0,
    )
    return VietnamStockEnv(
        data,
        start_date=str(data.dates[2].date()),
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
    )


def action_for(env: VietnamStockEnv, *weights: float) -> np.ndarray:
    return env.weights_to_action(np.asarray(weights, dtype=np.float64))


def test_observation_uses_only_information_before_execution_day() -> None:
    env = make_env(n_assets=1)
    observation = env.reset()
    asset_history = observation[: env.spec.asset_history_size].reshape(
        1, env.lookback, len(STOCK_FEATURE_NAMES)
    )
    assert asset_history[0, :, 0].tolist() == [0.0, 1.0]
    assert env.current_index == 2


def test_observation_uses_ex_ante_decision_mask() -> None:
    env = make_env(n_assets=1)
    env.data.tradable_mask[env.period_start, 0] = False
    observation = env.reset()
    portfolio_offset = env.spec.asset_history_size + env.spec.market_history_size
    assert observation[portfolio_offset] == 0.0


def test_execution_price_does_not_use_same_day_high_or_low() -> None:
    env = make_env(n_assets=1)
    env.reset()
    index = env.current_index
    before = env._execution_prices(index)
    env.data.high_prices[index, 0] = 1_000_000.0
    env.data.low_prices[index, 0] = 0.001
    after = env._execution_prices(index)
    assert np.allclose(before[0], after[0])
    assert np.allclose(before[1], after[1])


def test_execution_volume_cap_uses_previous_completed_session() -> None:
    env = make_env(n_assets=1)
    env.reset()
    index = env.current_index
    env.data.volumes[index - 1, 0] = 1_000.0
    env.data.volumes[index, 0] = 1_000_000.0
    _, _, _, info = env.step(action_for(env, 1.0, 0.0))
    assert info["buy_shares"][0] == 1_000


def test_settlement_delayed_initial_cash_cannot_trade_early() -> None:
    data = synthetic_bundle(n_assets=1)
    config = MarketConfig(
        initial_cash=100_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=0.0,
        max_volume_fraction=1.0,
    )
    env = VietnamStockEnv(
        data,
        start_date=str(data.dates[2].date()),
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
        initial_cash_delay_days=2,
    )
    env.reset()
    stock = action_for(env, 1.0, 0.0)
    assert env.step(stock)[3]["buy_notional"] == 0.0
    assert env.step(stock)[3]["buy_notional"] == 0.0
    assert env.step(stock)[3]["buy_notional"] > 0.0


def test_rejected_target_does_not_receive_fictitious_allocation_alpha() -> None:
    data = synthetic_bundle(n_assets=2)
    data.close_prices[2] = [11.0, 9.0]
    data.valuation_prices[2] = [11.0, 9.0]
    config = MarketConfig(
        initial_cash=100_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=0.0,
        max_volume_fraction=1.0,
        allocation_alpha_reward=1.0,
    )
    env = VietnamStockEnv(
        data,
        start_date=str(data.dates[2].date()),
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
        initial_cash_delay_days=2,
    )
    env.reset()
    _, _, _, info = env.step(action_for(env, 1.0, 0.0, 0.0))
    assert info["buy_notional"] == 0.0
    assert info["allocation_alpha"] == 0.0
    assert info["allocation_alpha_reward"] == 0.0


def test_missing_open_does_not_mark_an_existing_position_to_zero() -> None:
    env = make_env(n_assets=1)
    env.reset()
    stock = action_for(env, 1.0, 0.0)
    env.step(stock)
    env.step(stock)
    index = env.current_index
    env.data.open_prices[index, 0] = np.nan
    env.data.execution_mask[index, 0] = False
    _, _, _, info = env.step(action_for(env, 0.0, 1.0))
    assert info["sell_notional"] == 0.0
    assert np.isclose(info["pretrade_nav"], 100_000.0)
    assert np.isclose(info["nav"], 100_000.0)
    assert np.isfinite(info["slippage_cost"])


def test_slippage_is_reported_as_a_transaction_cost() -> None:
    data = synthetic_bundle(n_assets=1)
    config = MarketConfig(
        initial_cash=100_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=100.0,
        max_volume_fraction=1.0,
    )
    env = VietnamStockEnv(
        data,
        start_date=str(data.dates[2].date()),
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
    )
    env.reset()
    _, _, _, info = env.step(action_for(env, 1.0, 0.0))
    assert info["buy_shares"][0] == 9_900
    assert np.isclose(info["slippage_cost"], 990.0)
    assert np.isclose(info["nav"], 99_010.0)


def test_historical_hose_round_lot_changes_from_10_to_100() -> None:
    data = synthetic_bundle(n_assets=1)
    data.dates = pd.bdate_range("2020-12-29", periods=data.n_dates)
    config = MarketConfig(
        initial_cash=1_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        legacy_lot_size=10,
        lot_size_100_effective_date="2021-01-04",
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=0.0,
        max_volume_fraction=1.0,
        minimum_trade_fraction=0.0,
    )
    env = VietnamStockEnv(
        data,
        start_date="2020-12-31",
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
    )
    env.reset()
    target = action_for(env, 0.15, 0.85)
    _, _, _, before = env.step(target)
    assert before["buy_shares"][0] == 10
    env.step(action_for(env, 0.0, 1.0))
    assert env._lot_size(env.current_index) == 100
    _, _, _, transition_day = env.step(action_for(env, 0.0, 1.0))
    assert transition_day["sell_shares"][0] == 0
    assert env.available_shares[0] == 10


def test_hose_odd_lot_board_can_sell_residual_shares_from_2022_09_12() -> None:
    data = synthetic_bundle(n_assets=1)
    data.dates = pd.bdate_range("2022-09-08", periods=data.n_dates)
    config = MarketConfig(
        initial_cash=100_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        legacy_lot_size=10,
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=0.0,
        max_volume_fraction=1.0,
        minimum_trade_fraction=0.0,
    )
    env = VietnamStockEnv(
        data,
        start_date="2022-09-12",
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
    )
    env.reset()
    env.available_shares[0] = 50
    env.available_cash = 99_500.0

    _, _, _, info = env.step(action_for(env, 0.0, 1.0))

    assert info["sell_shares"][0] == 50
    assert env.available_shares[0] == 0


def test_bought_shares_are_not_sellable_until_t_plus_two() -> None:
    env = make_env(n_assets=1)
    env.reset()
    stock, cash = action_for(env, 1.0, 0.0), action_for(env, 0.0, 1.0)

    _, _, _, first = env.step(stock)
    assert first["buy_shares"][0] == 10_000
    assert env.available_shares[0] == 0
    assert env._pending_share_total()[0] == 10_000

    _, _, _, second = env.step(cash)
    assert second["sell_shares"][0] == 0
    assert env.available_shares[0] == 10_000  # settled before the next action at T+2

    _, _, _, third = env.step(cash)
    assert third["sell_shares"][0] == 10_000
    assert env.available_shares[0] == 0
    assert env._pending_cash_total() == 100_000


def test_sale_proceeds_cannot_fund_purchase_until_t_plus_two() -> None:
    env = make_env(n_assets=2)
    env.reset()
    asset_a = action_for(env, 1.0, 0.0, 0.0)
    asset_b = action_for(env, 0.0, 1.0, 0.0)

    env.step(asset_a)
    env.step(asset_a)  # A settles before the following decision.
    _, _, _, sale_day = env.step(asset_b)
    assert sale_day["sell_shares"][0] == 10_000
    assert sale_day["buy_shares"][1] == 0
    assert env.available_cash == 0

    _, _, _, next_day = env.step(asset_b)
    assert next_day["buy_shares"][1] == 0
    assert env.available_cash == 100_000  # proceeds settle for next decision

    _, _, _, settlement_day = env.step(asset_b)
    assert settlement_day["buy_shares"][1] == 10_000


def test_nav_includes_pending_shares_and_pending_sale_cash() -> None:
    env = make_env(n_assets=1)
    env.reset()
    stock, cash = action_for(env, 1.0, 0.0), action_for(env, 0.0, 1.0)
    _, _, _, buy_info = env.step(stock)
    assert buy_info["nav"] == 100_000
    env.step(cash)
    _, _, _, sell_info = env.step(cash)
    assert sell_info["nav"] == 100_000
    assert sell_info["pending_sale_cash"] == 100_000


def test_no_trade_band_suppresses_small_rebalance_orders() -> None:
    data = synthetic_bundle(n_assets=2)
    config = MarketConfig(
        initial_cash=100_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=0.0,
        max_volume_fraction=1.0,
        minimum_trade_fraction=0.02,
    )
    env = VietnamStockEnv(
        data,
        start_date=str(data.dates[2].date()),
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
    )
    env.reset()
    equal_weight = action_for(env, 0.5, 0.5, 0.0)
    env.step(equal_weight)
    env.step(equal_weight)
    small_shift = action_for(env, 0.51, 0.49, 0.0)
    _, _, _, info = env.step(small_shift)
    assert info["buy_notional"] == 0
    assert info["sell_notional"] == 0


def test_explicit_hold_does_not_rebalance_an_overnight_gap() -> None:
    env = make_env(n_assets=1)
    env.reset()
    half_stock = action_for(env, 0.5, 0.5)
    env.step(half_stock)
    env.step(None)  # Shares settle before the next decision.
    index = env.current_index
    env.data.open_prices[index, 0] = 20.0
    env.data.close_prices[index, 0] = 20.0
    env.data.valuation_prices[index, 0] = 20.0
    shares_before = env.available_shares.copy()

    _, _, _, info = env.step(None)

    assert np.array_equal(env.available_shares, shares_before)
    assert info["buy_notional"] == 0.0
    assert info["sell_notional"] == 0.0
    assert info["turnover"] == 0.0


def test_current_constituent_position_is_frozen_when_decision_data_is_missing() -> None:
    env = make_env(n_assets=1)
    env.reset()
    stock = action_for(env, 1.0, 0.0)
    env.step(stock)
    env.step(None)  # Shares settle before the next decision.
    index = env.current_index
    env.data.tradable_mask[index, 0] = False
    assert env.data.universe_mask[index, 0]

    _, _, _, info = env.step(action_for(env, 0.0, 1.0))

    assert info["sell_notional"] == 0.0
    assert env.available_shares[0] > 0.0


def test_training_reward_can_include_cross_sectional_allocation_alpha() -> None:
    data = synthetic_bundle(n_assets=2)
    data.close_prices[2] = [11.0, 9.0]
    data.valuation_prices[2] = [11.0, 9.0]
    config = MarketConfig(
        initial_cash=100_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=0.0,
        max_volume_fraction=1.0,
        turnover_penalty_rate=0.0,
        drawdown_penalty=0.0,
        allocation_alpha_reward=1.0,
    )
    env = VietnamStockEnv(
        data,
        start_date=str(data.dates[2].date()),
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
    )
    env.reset()
    _, reward, _, info = env.step(action_for(env, 1.0, 0.0, 0.0))
    assert np.isclose(info["allocation_alpha"], 0.1)
    assert np.isclose(info["allocation_alpha_reward"], 0.1)
    assert reward > 0


def test_allocation_alpha_is_scaled_by_realized_stock_exposure() -> None:
    data = synthetic_bundle(n_assets=2)
    data.close_prices[2] = [11.0, 9.0]
    data.valuation_prices[2] = [11.0, 9.0]
    config = MarketConfig(
        initial_cash=100_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        broker_fee_rate=0.0,
        sell_tax_rate=0.0,
        slippage_bps=0.0,
        max_volume_fraction=1.0,
        turnover_penalty_rate=0.0,
        drawdown_penalty=0.0,
        allocation_alpha_reward=1.0,
    )
    env = VietnamStockEnv(
        data,
        start_date=str(data.dates[2].date()),
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
    )
    env.reset()
    _, _, _, info = env.step(action_for(env, 0.10, 0.0, 0.90))
    assert np.isclose(info["allocation_alpha"], 0.01)


def test_random_actions_preserve_daily_accounting_identity() -> None:
    data = synthetic_bundle(n_assets=3, n_dates=30)
    rng = np.random.default_rng(123)
    open_returns = rng.normal(0.0, 0.01, size=(30, 3))
    close_returns = rng.normal(0.0, 0.01, size=(30, 3))
    opens = 10.0 * np.exp(np.cumsum(open_returns, axis=0))
    closes = opens * np.exp(close_returns)
    data.open_prices[:] = opens
    data.close_prices[:] = closes
    data.valuation_prices[:] = closes
    data.high_prices[:] = np.maximum(opens, closes) * 1.01
    data.low_prices[:] = np.minimum(opens, closes) * 0.99
    config = MarketConfig(
        initial_cash=1_000_000.0,
        settlement_days=2,
        price_multiplier=1.0,
        lot_size=100,
        broker_fee_rate=0.0015,
        sell_tax_rate=0.001,
        slippage_bps=5.0,
        max_volume_fraction=0.10,
    )
    env = VietnamStockEnv(
        data,
        start_date=str(data.dates[2].date()),
        end_date=str(data.dates[-1].date()),
        lookback=2,
        config=config,
    )
    env.reset()
    done = False
    while not done:
        previous_nav = env.last_nav
        executed_index = env.current_index
        logits = rng.normal(size=env.action_dim)
        _, _, done, info = env.step(logits)
        shares = env.available_shares + env._pending_share_total()
        accounting_nav = (
            env.available_cash
            + env._pending_cash_total()
            + float(np.dot(shares, env._mark_prices(executed_index)))
        )
        assert np.isclose(info["nav"], accounting_nav, rtol=1e-10, atol=1e-6)
        assert np.isclose(
            info["daily_return"], info["nav"] / previous_nav - 1.0
        )
        assert env.available_cash >= -1e-6
        assert (env.available_shares >= -1e-6).all()
        assert (env._pending_share_total() >= -1e-6).all()
        assert env._pending_cash_total() >= -1e-6
