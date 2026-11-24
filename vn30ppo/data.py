from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


STOCK_FEATURE_NAMES = (
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "overnight_gap",
    "intraday_return",
    "high_low_range",
    "volume_z20",
    "volatility_20d",
    "trend_20d",
    "drawdown_60d",
    "relative_strength_20d",
)

MARKET_FEATURE_NAMES = tuple(
    name for name in STOCK_FEATURE_NAMES if name != "relative_strength_20d"
)

FEATURE_HISTORY_OBSERVATIONS = 61
MINIMUM_ACTIVE_PRICE_COVERAGE = 0.80


@dataclass(slots=True)
class MarketDataBundle:
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    open_prices: np.ndarray
    high_prices: np.ndarray
    low_prices: np.ndarray
    close_prices: np.ndarray
    valuation_prices: np.ndarray
    volumes: np.ndarray
    stock_features: np.ndarray
    market_features: np.ndarray
    tradable_mask: np.ndarray
    execution_mask: np.ndarray
    universe_mask: np.ndarray
    market_open: np.ndarray
    market_close: np.ndarray
    eligible_assets: np.ndarray
    feature_names: tuple[str, ...] = STOCK_FEATURE_NAMES
    market_feature_names: tuple[str, ...] = MARKET_FEATURE_NAMES

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    def period_indices(self, start: str, end: str) -> np.ndarray:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        indices = np.flatnonzero((self.dates >= start_ts) & (self.dates <= end_ts))
        if len(indices) == 0:
            raise ValueError(f"No trading dates in requested period {start}..{end}")
        return indices


def _validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close", "volume", "ticker"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    result = frame.copy()
    result["time"] = pd.to_datetime(result["time"], errors="raise")
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values(["time", "ticker"]).reset_index(drop=True)


def _normalize_ohlc_envelope(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize adjusted vendor OHLC so high/low enclose open and close."""
    result = frame.copy()
    original_high = result["high"].copy()
    original_low = result["low"].copy()
    result["high"] = result[["open", "high", "close"]].max(axis=1)
    result["low"] = result[["open", "low", "close"]].min(axis=1)
    repaired = (result["high"] != original_high) | (result["low"] != original_low)
    if repaired.any():
        warnings.warn(
            f"Normalized OHLC envelope on {int(repaired.sum())} adjusted provider rows",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def _validate_selected_ohlcv(frame: pd.DataFrame) -> None:
    """Validate only VNINDEX and point-in-time constituent-union rows."""

    if frame.duplicated(["time", "ticker"]).any():
        duplicates = int(frame.duplicated(["time", "ticker"]).sum())
        raise ValueError(f"Input contains {duplicates} duplicate date/ticker rows")
    price_columns = ["open", "high", "low", "close"]
    nonpositive_prices = frame[price_columns].le(0) & frame[price_columns].notna()
    if nonpositive_prices.any(axis=None):
        raise ValueError("Input contains non-positive OHLC prices")
    if ((frame["volume"] < 0) & frame["volume"].notna()).any():
        raise ValueError("Input contains negative volume")
    complete_ohlc = frame[price_columns].notna().all(axis=1)
    inconsistent = complete_ohlc & (
        (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame["low"])
    )
    if inconsistent.any():
        example = frame.loc[
            inconsistent, ["time", "ticker", *price_columns]
        ].iloc[0]
        raise ValueError(
            "Input contains inconsistent OHLC values, for example "
            f"{example['ticker']} on {example['time'].date()}"
        )


def _load_membership_table(path: str | Path) -> pd.DataFrame:
    membership_path = Path(path)
    if not membership_path.exists():
        raise FileNotFoundError(
            f"Missing point-in-time VN30 membership file: {membership_path}. "
            "Expected columns: effective_from,effective_to,ticker,source. "
            "Do not substitute the current VN30 list for historical membership."
        )
    table = pd.read_csv(membership_path)
    required = {"effective_from", "effective_to", "ticker", "source"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Membership file is missing columns: {sorted(missing)}"
        )
    table = table.loc[:, sorted(required)].copy()
    table["effective_from"] = pd.to_datetime(
        table["effective_from"], errors="raise"
    )
    table["effective_to"] = pd.to_datetime(table["effective_to"], errors="raise")
    table["ticker"] = table["ticker"].astype(str).str.upper().str.strip()
    table["source"] = table["source"].astype(str).str.strip()
    if table.empty:
        raise ValueError("Membership file cannot be empty")
    if (table["ticker"] == "VNINDEX").any() or (table["ticker"] == "").any():
        raise ValueError("Membership tickers must be non-empty stocks, not VNINDEX")
    if (table["source"] == "").any():
        raise ValueError("Every membership interval must include an auditable source")
    if (table["effective_from"] > table["effective_to"]).any():
        raise ValueError("Membership effective_from must be <= effective_to")
    if table.duplicated(["effective_from", "effective_to", "ticker"]).any():
        raise ValueError("Membership file contains duplicate intervals")
    return table.sort_values(["ticker", "effective_from"]).reset_index(drop=True)


def _load_static_universe(path: str | Path, expected_size: int) -> pd.DataFrame:
    universe_path = Path(path)
    if not universe_path.exists():
        raise FileNotFoundError(
            f"Missing current VN30 universe file: {universe_path}. "
            "Run `python cao_data.py` to create it."
        )
    raw = pd.read_csv(universe_path)
    price_columns = {"time", "open", "high", "low", "close", "volume", "ticker"}
    if price_columns.issubset(raw.columns):
        tickers = sorted(
            set(raw["ticker"].astype(str).str.upper().str.strip()) - {"VNINDEX"}
        )
        table = pd.DataFrame(
            {
                "ticker": tickers,
                "as_of": pd.to_datetime(raw["time"], errors="raise").max(),
                "source": f"ticker snapshot embedded in {universe_path.name}",
            }
        )
    else:
        required = {"ticker", "as_of", "source"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(
                f"Static universe file is missing columns: {sorted(missing)}"
            )
        table = raw.loc[:, ["ticker", "as_of", "source"]].copy()
    table["ticker"] = table["ticker"].astype(str).str.upper().str.strip()
    table["as_of"] = pd.to_datetime(table["as_of"], errors="raise")
    table["source"] = table["source"].astype(str).str.strip()
    if table["ticker"].duplicated().any():
        raise ValueError("Static universe file contains duplicate tickers")
    if (table["ticker"] == "").any() or (table["ticker"] == "VNINDEX").any():
        raise ValueError("Static universe tickers must be non-empty stocks, not VNINDEX")
    if (table["source"] == "").any():
        raise ValueError("Static universe rows must include a source")
    if len(table) != expected_size:
        raise ValueError(
            f"Static universe must contain exactly {expected_size} tickers; "
            f"received {len(table)}"
        )
    return table.sort_values("ticker").reset_index(drop=True)


def _build_universe_mask(
    membership: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    tickers: tuple[str, ...],
    *,
    required_start: str,
    required_end: str,
    expected_size: int,
) -> np.ndarray:
    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
    mask = np.zeros((len(calendar), len(tickers)), dtype=bool)
    for row in membership.itertuples(index=False):
        asset_index = ticker_index[row.ticker]
        active = (calendar >= row.effective_from) & (calendar <= row.effective_to)
        if np.any(mask[active, asset_index]):
            raise ValueError(
                f"Overlapping membership intervals found for {row.ticker}"
            )
        mask[active, asset_index] = True

    required = (calendar >= pd.Timestamp(required_start)) & (
        calendar <= pd.Timestamp(required_end)
    )
    if not required.any():
        raise ValueError("Required experiment period does not overlap market dates")
    counts = mask[required].sum(axis=1)
    invalid = np.flatnonzero(counts != expected_size)
    if len(invalid):
        required_dates = calendar[required]
        examples = ", ".join(
            f"{required_dates[index].date()}={int(counts[index])}"
            for index in invalid[:5]
        )
        raise ValueError(
            "Point-in-time universe must contain exactly "
            f"{expected_size} members on every required trading day; {examples}"
        )
    return mask


def _feature_frame(raw: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    values = raw.set_index("time").reindex(calendar)
    close = values["close"].where(values["close"] > 0)
    open_price = values["open"].where(values["open"] > 0)
    high = values["high"].where(values["high"] > 0)
    low = values["low"].where(values["low"] > 0)
    log_close = np.log(close)
    return_1d = log_close.diff(1)
    log_volume = np.log1p(values["volume"].clip(lower=0))
    volume_mean = log_volume.rolling(20, min_periods=10).mean()
    volume_std = log_volume.rolling(20, min_periods=10).std().replace(0, np.nan)

    features = pd.DataFrame(index=calendar)
    features["return_1d"] = return_1d
    features["return_5d"] = log_close.diff(5)
    features["return_20d"] = log_close.diff(20)
    features["return_60d"] = log_close.diff(60)
    features["overnight_gap"] = np.log(open_price / close.shift(1))
    features["intraday_return"] = np.log(close / open_price)
    features["high_low_range"] = np.log(high / low)
    features["volume_z20"] = (log_volume - volume_mean) / volume_std
    features["volatility_20d"] = return_1d.rolling(20, min_periods=10).std() * np.sqrt(252)
    features["trend_20d"] = np.log(
        close / close.rolling(20, min_periods=10).mean()
    )
    features["drawdown_60d"] = (
        close / close.rolling(60, min_periods=20).max() - 1.0
    )
    return features.replace([np.inf, -np.inf], np.nan)


def _standardize_panel(
    panel: np.ndarray,
    training_rows: np.ndarray,
    training_asset_mask: np.ndarray | None = None,
) -> np.ndarray:
    result = panel.copy()
    training = result[training_rows]
    if training_asset_mask is not None:
        if training_asset_mask.shape != training.shape[:2]:
            raise ValueError("training_asset_mask has an invalid shape")
        training = np.where(training_asset_mask[..., None], training, np.nan)
    means, stds = _finite_mean_std(training, axis=(0, 1))
    stds = np.where(stds < 1e-8, 1.0, stds)
    result = (result - means[None, None, :]) / stds[None, None, :]
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10, 10)


def _standardize_market(
    features: np.ndarray,
    training_rows: np.ndarray,
) -> np.ndarray:
    training = features[training_rows]
    means, stds = _finite_mean_std(training, axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    result = (features - means[None, :]) / stds[None, :]
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10, 10)


def _finite_mean_std(
    values: np.ndarray, axis: int | tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite-only moments without emitting empty-slice warnings."""

    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    totals = np.where(finite, values, 0.0).sum(axis=axis)
    means = np.divide(
        totals,
        counts,
        out=np.zeros_like(totals, dtype=np.float64),
        where=counts > 0,
    )
    if isinstance(axis, tuple):
        expanded = means
        for item in sorted(axis):
            expanded = np.expand_dims(expanded, axis=item)
    else:
        expanded = np.expand_dims(means, axis=axis)
    squared = np.where(finite, (values - expanded) ** 2, 0.0).sum(axis=axis)
    variances = np.divide(
        squared,
        counts,
        out=np.zeros_like(squared, dtype=np.float64),
        where=counts > 0,
    )
    return means, np.sqrt(variances)


def load_market_data(
    path: str | Path,
    *,
    train_start: str,
    train_end: str,
    lookback: int = 20,
    minimum_training_days: int = 252,
    membership_path: str | Path | None = None,
    require_point_in_time_universe: bool = True,
    required_end: str | None = None,
    expected_universe_size: int = 30,
) -> MarketDataBundle:
    """Load VN30 data and fit every normalization statistic on training only.

    A point-in-time membership table remains supported for reproducibility, but
    the default research configuration uses a static snapshot of today's VN30.
    A current constituent that listed later becomes tradable only after enough
    public price history exists.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = _normalize_ohlc_envelope(_validate_input(pd.read_csv(path)))
    market_rows = frame[frame["ticker"] == "VNINDEX"].copy()
    all_stock_rows = frame[frame["ticker"] != "VNINDEX"].copy()
    membership: pd.DataFrame | None = None
    static_universe: pd.DataFrame | None = None
    if membership_path is not None:
        universe_columns = set(pd.read_csv(membership_path, nrows=0).columns)
        if {"effective_from", "effective_to", "ticker", "source"}.issubset(
            universe_columns
        ):
            membership = _load_membership_table(membership_path)
            selected_tickers = set(membership["ticker"])
        else:
            static_universe = _load_static_universe(
                membership_path, expected_universe_size
            )
            selected_tickers = set(static_universe["ticker"])
        data_tickers = set(all_stock_rows["ticker"])
        missing_prices = sorted(selected_tickers.difference(data_tickers))
        if missing_prices:
            raise ValueError(
                "Price dataset is missing required VN30 constituents: "
                + ", ".join(missing_prices)
            )
        stock_rows = all_stock_rows[
            all_stock_rows["ticker"].isin(selected_tickers)
        ].copy()
    elif require_point_in_time_universe:
        raise ValueError(
            "membership_path is required; a static/current VN30 universe would "
            "introduce survivorship and constituent look-ahead bias"
        )
    else:
        warnings.warn(
            "Using every stock in the price file as a static universe. Results "
            "are not valid as a historical VN30 backtest.",
            RuntimeWarning,
            stacklevel=2,
        )
        stock_rows = all_stock_rows
    _validate_selected_ohlcv(pd.concat([market_rows, stock_rows], ignore_index=True))
    if market_rows.empty or stock_rows.empty:
        raise ValueError("Both VNINDEX and stock rows are required")

    calendar = pd.DatetimeIndex(market_rows["time"].drop_duplicates().sort_values())
    tickers = tuple(sorted(stock_rows["ticker"].unique()))
    n_dates, n_assets = len(calendar), len(tickers)
    shape = (n_dates, n_assets)
    opens = np.full(shape, np.nan, dtype=np.float64)
    highs = np.full(shape, np.nan, dtype=np.float64)
    lows = np.full(shape, np.nan, dtype=np.float64)
    closes = np.full(shape, np.nan, dtype=np.float64)
    volumes = np.zeros(shape, dtype=np.float64)
    feature_panel = np.full(
        (n_dates, n_assets, len(STOCK_FEATURE_NAMES)), np.nan, dtype=np.float64
    )

    training_date_mask = (calendar >= pd.Timestamp(train_start)) & (
        calendar <= pd.Timestamp(train_end)
    )
    training_rows = np.flatnonzero(training_date_mask)
    if len(training_rows) == 0:
        raise ValueError("Training period does not overlap the dataset")

    if membership is not None:
        universe_mask = _build_universe_mask(
            membership,
            calendar,
            tickers,
            required_start=train_start,
            required_end=required_end or str(calendar.max().date()),
            expected_size=expected_universe_size,
        )
    else:
        universe_mask = np.ones(shape, dtype=bool)

    market_feature_frame = _feature_frame(market_rows, calendar).reindex(
        columns=MARKET_FEATURE_NAMES
    )
    training_counts = np.zeros(n_assets, dtype=np.int64)
    history_counts = np.zeros(shape, dtype=np.int64)
    required_date_mask = (calendar >= pd.Timestamp(train_start)) & (
        calendar <= pd.Timestamp(required_end or str(calendar.max().date()))
    )
    insufficient_coverage: list[str] = []
    for asset_index, ticker in enumerate(tickers):
        raw = stock_rows[stock_rows["ticker"] == ticker].copy()
        indexed = raw.set_index("time").reindex(calendar)
        opens[:, asset_index] = indexed["open"].to_numpy(dtype=np.float64)
        highs[:, asset_index] = indexed["high"].to_numpy(dtype=np.float64)
        lows[:, asset_index] = indexed["low"].to_numpy(dtype=np.float64)
        closes[:, asset_index] = indexed["close"].to_numpy(dtype=np.float64)
        volumes[:, asset_index] = indexed["volume"].fillna(0).to_numpy(dtype=np.float64)
        stock_feature_frame = _feature_frame(raw, calendar)
        stock_feature_frame["relative_strength_20d"] = (
            stock_feature_frame["return_20d"]
            - market_feature_frame["return_20d"]
        )
        feature_panel[:, asset_index] = stock_feature_frame.reindex(
            columns=STOCK_FEATURE_NAMES
        ).to_numpy(dtype=np.float64)
        observed = np.isfinite(closes[:, asset_index])
        history_counts[:, asset_index] = np.cumsum(observed)
        training_counts[asset_index] = int(
            (observed & universe_mask[:, asset_index] & training_date_mask).sum()
        )
        active = universe_mask[:, asset_index] & required_date_mask
        if static_universe is not None:
            observed_dates = calendar[observed & required_date_mask]
            if len(observed_dates):
                active &= (calendar >= observed_dates.min()) & (
                    calendar <= observed_dates.max()
                )
        if active.any():
            opening_and_close_observed = (
                np.isfinite(opens[:, asset_index])
                & (opens[:, asset_index] > 0)
                & np.isfinite(closes[:, asset_index])
                & (closes[:, asset_index] > 0)
            )
            coverage = float(opening_and_close_observed[active].mean())
            if coverage < MINIMUM_ACTIVE_PRICE_COVERAGE:
                insufficient_coverage.append(f"{ticker}={coverage:.1%}")

    if insufficient_coverage:
        raise ValueError(
            "VN30 constituents have insufficient active-period OHLC "
            f"coverage (minimum {MINIMUM_ACTIVE_PRICE_COVERAGE:.0%}): "
            + ", ".join(insufficient_coverage)
        )

    eligible = training_counts >= minimum_training_days
    raw_observed = (
        np.isfinite(opens)
        & np.isfinite(closes)
        & (opens > 0)
        & (closes > 0)
        & (volumes > 0)
    )
    previous_observed = np.zeros_like(raw_observed)
    previous_observed[1:] = raw_observed[:-1]
    previous_history = np.zeros_like(history_counts)
    previous_history[1:] = history_counts[:-1]
    decision_mask = (
        universe_mask
        & previous_observed
        & (
            previous_history
            >= max(lookback, FEATURE_HISTORY_OBSERVATIONS)
        )
    )
    execution_mask = np.isfinite(opens) & (opens > 0)

    market_indexed = market_rows.set_index("time").reindex(calendar)
    market_open = market_indexed["open"].to_numpy(dtype=np.float64)
    market_close = market_indexed["close"].to_numpy(dtype=np.float64)
    valid_market = (
        np.isfinite(market_open)
        & (market_open > 0)
        & np.isfinite(market_close)
        & (market_close > 0)
    )
    if not valid_market.all():
        invalid_dates = ", ".join(
            str(date.date()) for date in calendar[~valid_market][:5]
        )
        raise ValueError(f"VNINDEX has invalid open/close values on: {invalid_dates}")
    market_features = market_feature_frame.to_numpy(dtype=np.float64)

    feature_panel = _standardize_panel(
        feature_panel,
        training_rows,
        universe_mask[training_rows],
    ).astype(np.float32)
    market_features = _standardize_market(market_features, training_rows).astype(np.float32)
    valuation_prices = pd.DataFrame(closes, index=calendar).ffill().fillna(0).to_numpy()

    return MarketDataBundle(
        dates=calendar,
        tickers=tickers,
        open_prices=opens,
        high_prices=highs,
        low_prices=lows,
        close_prices=closes,
        valuation_prices=valuation_prices,
        volumes=volumes,
        stock_features=feature_panel,
        market_features=market_features,
        tradable_mask=decision_mask,
        execution_mask=execution_mask,
        universe_mask=universe_mask,
        market_open=market_open,
        market_close=market_close,
        eligible_assets=eligible,
    )
