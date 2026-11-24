from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vn30ppo.data import load_market_data


def _write_prices(path: Path) -> pd.DatetimeIndex:
    dates = pd.bdate_range("2024-01-02", periods=75)
    rows: list[dict[str, object]] = []
    for ticker in ("A", "B", "VNINDEX"):
        for offset, date in enumerate(dates):
            price = 10.0 + offset * 0.1
            rows.append(
                {
                    "time": date,
                    "ticker": ticker,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1_000_000.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return dates


def _write_membership(path: Path, dates: pd.DatetimeIndex) -> None:
    pd.DataFrame(
        [
            {
                "effective_from": dates[0],
                "effective_to": dates[39],
                "ticker": "A",
                "source": "official-test-source-a",
            },
            {
                "effective_from": dates[40],
                "effective_to": dates[-1],
                "ticker": "B",
                "source": "official-test-source-b",
            },
        ]
    ).to_csv(path, index=False)


def test_point_in_time_membership_changes_the_daily_universe(tmp_path: Path) -> None:
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    dates = _write_prices(prices)
    _write_membership(membership, dates)
    data = load_market_data(
        prices,
        train_start=str(dates[0].date()),
        train_end=str(dates[39].date()),
        lookback=2,
        minimum_training_days=1,
        membership_path=membership,
        required_end=str(dates[-1].date()),
        expected_universe_size=1,
    )
    a, b = (data.tickers.index(ticker) for ticker in ("A", "B"))
    assert data.universe_mask[0, a]
    assert not data.universe_mask[0, b]
    assert not data.universe_mask[-1, a]
    assert data.universe_mask[-1, b]
    assert not data.eligible_assets[b]
    assert data.tradable_mask[-1, b]


def test_static_current_universe_selects_only_snapshot_tickers(tmp_path: Path) -> None:
    prices = tmp_path / "prices.csv"
    dates = _write_prices(prices)
    universe = tmp_path / "current.csv"
    pd.DataFrame(
        [{"ticker": "A", "as_of": "2026-08-27", "source": "test-provider"}]
    ).to_csv(universe, index=False)
    data = load_market_data(
        prices,
        train_start=str(dates[0].date()),
        train_end=str(dates[39].date()),
        lookback=2,
        minimum_training_days=1,
        membership_path=universe,
        require_point_in_time_universe=False,
        required_end=str(dates[-1].date()),
        expected_universe_size=1,
    )
    assert data.tickers == ("A",)
    assert data.universe_mask.all()


def test_price_dataset_can_supply_the_static_current_universe(tmp_path: Path) -> None:
    prices = tmp_path / "prices.csv"
    dates = _write_prices(prices)
    data = load_market_data(
        prices,
        train_start=str(dates[0].date()),
        train_end=str(dates[39].date()),
        lookback=2,
        minimum_training_days=1,
        membership_path=prices,
        require_point_in_time_universe=False,
        required_end=str(dates[-1].date()),
        expected_universe_size=2,
    )
    assert data.tickers == ("A", "B")
    assert data.universe_mask.all()


def test_missing_historical_constituent_prices_are_rejected(tmp_path: Path) -> None:
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    dates = _write_prices(prices)
    pd.DataFrame(
        [
            {
                "effective_from": dates[0],
                "effective_to": dates[-1],
                "ticker": "C",
                "source": "official-test-source-c",
            }
        ]
    ).to_csv(membership, index=False)
    with pytest.raises(ValueError, match="missing required VN30 constituents: C"):
        load_market_data(
            prices,
            train_start=str(dates[0].date()),
            train_end=str(dates[39].date()),
            lookback=2,
            minimum_training_days=1,
            membership_path=membership,
            required_end=str(dates[-1].date()),
            expected_universe_size=1,
        )


def test_ticker_present_only_outside_membership_is_rejected(tmp_path: Path) -> None:
    prices = tmp_path / "prices.csv"
    membership = tmp_path / "membership.csv"
    dates = _write_prices(prices)
    frame = pd.read_csv(prices, parse_dates=["time"])
    frame = frame[~((frame["ticker"] == "B") & (frame["time"] >= dates[40]))]
    frame.to_csv(prices, index=False)
    pd.DataFrame(
        [
            {
                "effective_from": dates[40],
                "effective_to": dates[-1],
                "ticker": "B",
                "source": "official-test-source-b",
            }
        ]
    ).to_csv(membership, index=False)
    with pytest.raises(ValueError, match="insufficient active-period OHLC coverage"):
        load_market_data(
            prices,
            train_start=str(dates[40].date()),
            train_end=str(dates[60].date()),
            lookback=2,
            minimum_training_days=1,
            membership_path=membership,
            required_end=str(dates[-1].date()),
            expected_universe_size=1,
        )


def test_adjusted_ohlc_envelope_is_normalized(tmp_path: Path) -> None:
    prices = tmp_path / "prices.csv"
    dates = _write_prices(prices)
    frame = pd.read_csv(prices)
    frame.loc[0, "high"] = frame.loc[0, "low"] - 1.0
    frame.to_csv(prices, index=False)
    with pytest.warns(RuntimeWarning, match="Normalized OHLC envelope"):
        data = load_market_data(
            prices,
            train_start=str(dates[0].date()),
            train_end=str(dates[39].date()),
            lookback=2,
            minimum_training_days=1,
            membership_path=prices,
            require_point_in_time_universe=False,
            required_end=str(dates[-1].date()),
            expected_universe_size=2,
        )
    assert data.high_prices[0, 0] >= data.open_prices[0, 0]
    assert data.high_prices[0, 0] >= data.close_prices[0, 0]
    assert data.low_prices[0, 0] <= data.open_prices[0, 0]
    assert data.low_prices[0, 0] <= data.close_prices[0, 0]
