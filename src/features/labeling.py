"""Create ML target labels from forward returns."""

from __future__ import annotations

import pandas as pd


def create_labels(
    dataframe: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.005,
) -> pd.Series:
    """Create numeric BUY/HOLD/SELL labels from forward returns.

    1 = BUY (future return > threshold)
    0 = HOLD (future return within +/- threshold)
    -1 = SELL (future return < -threshold)

    The last `horizon` rows have no future data available and are
    labeled NaN. Does not mutate the input dataframe.
    """
    if "close" not in dataframe.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")

    close = dataframe["close"]
    future_return = (close.shift(-horizon) - close) / close

    labels = pd.Series(0.0, index=dataframe.index, name="label")
    labels[future_return > threshold] = 1.0
    labels[future_return < -threshold] = -1.0
    labels[future_return.isna()] = float("nan")

    return labels