"""Create ML target labels from forward returns."""

from __future__ import annotations

import pandas as pd


def create_labels(
    dataframe: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.005,
    volatility: pd.Series | None = None,
    volatility_multiplier: float = 1.5,
) -> pd.Series:
    """Create numeric BUY/HOLD/SELL labels from forward returns.

    1 = BUY (future return > threshold)
    0 = HOLD (future return within +/- threshold)
    -1 = SELL (future return < -threshold)

    By default the BUY/SELL boundary is a fixed `threshold` (e.g. 0.005 =
    0.5%) applied uniformly regardless of how volatile the market
    currently is. That fixed cutoff turns out to be several times the
    typical 5-bar move in calm periods and roughly equal to the typical
    move in volatile ones -- i.e. a much easier bar to clear in some
    regimes than others, which is not a real difference in signal.

    Pass `volatility` (e.g. ATR as a fraction of price, "atr_pct") to use
    a per-row boundary instead: `volatility_multiplier * volatility[t]` at
    each row t, so the BUY/SELL cutoff scales with the prevailing regime
    instead of being a fixed absolute number. `threshold` is ignored when
    `volatility` is given. `volatility` must use only current/past
    information (a trailing indicator like ATR, not a centered or
    forward-looking one) or this introduces label leakage.

    Rows where `volatility` is NaN (indicator warm-up) get a NaN label,
    the same treatment as the last `horizon` rows with no future data.

    Does not mutate the input dataframe.
    """
    if "close" not in dataframe.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")

    if volatility is not None and volatility_multiplier <= 0:
        raise ValueError("volatility_multiplier must be greater than 0")

    close = dataframe["close"]
    future_return = (close.shift(-horizon) - close) / close

    if volatility is not None:
        effective_threshold = volatility.reindex(dataframe.index) * volatility_multiplier
    else:
        effective_threshold = pd.Series(threshold, index=dataframe.index)

    labels = pd.Series(0.0, index=dataframe.index, name="label")
    labels[future_return > effective_threshold] = 1.0
    labels[future_return < -effective_threshold] = -1.0
    labels[future_return.isna() | effective_threshold.isna()] = float("nan")

    return labels