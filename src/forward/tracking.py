"""Pure helpers for the forward test: storing new bars, choosing what to predict, scoring outcomes.

The forward test is the only evaluation data nobody has tuned against: bars
that did not exist when the production model was trained. These helpers keep
it honest:

- Only fully closed bars are stored (a bar stamped 13:00 covers 13:00-14:00,
  so it is complete only once the clock passes 14:00). Twelve Data's newest
  bar is usually still forming.
- Storage is append-only: a bar already stored is never overwritten by a
  later fetch, so nothing evaluated earlier can silently change.
- A bar is only predicted once every secondary instrument also has data up to
  that bar, matching how the training data was built.
"""

from __future__ import annotations

import pandas as pd

from features.labeling import create_labels
from ml.dataset import CLASS_NAMES

BAR_LENGTH = pd.Timedelta(hours=1)


def complete_new_bars(
    fetched: pd.DataFrame,
    after: pd.Timestamp,
    now: pd.Timestamp,
    bar_length: pd.Timedelta = BAR_LENGTH,
) -> pd.DataFrame:
    """Bars in `fetched` strictly after `after` that had fully closed by `now`."""
    if fetched.empty:
        return fetched
    closed = fetched.index + bar_length <= now
    return fetched.loc[(fetched.index > after) & closed].sort_index()


def append_bars(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Append only bars whose timestamps are not stored yet; stored bars are never changed."""
    if existing is None or existing.empty:
        return new.sort_index()
    fresh = new.loc[~new.index.isin(existing.index)]
    return pd.concat([existing, fresh]).sort_index()


def bars_ready_to_predict(
    primary_index: pd.DatetimeIndex,
    secondary_last_bars: dict[str, pd.Timestamp],
    already_logged: pd.Index,
    start_after: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Primary bars after `start_after`, not yet logged, covered by every secondary's data."""
    covered_until = min(secondary_last_bars.values()) if secondary_last_bars else primary_index.max()
    mask = (primary_index > start_after) & (primary_index <= covered_until) & ~primary_index.isin(already_logged)
    return primary_index[mask]


def attach_outcomes(
    predictions: pd.DataFrame,
    primary: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    """Add what actually happened `horizon` bars after each prediction.

    Adds future_close, realized_return and actual (-1/0/1, the same labelling
    the model was trained on). These stay NaN until `horizon` later bars exist.
    """
    labels = create_labels(primary, horizon=horizon, threshold=threshold)
    future_close = primary["close"].shift(-horizon)
    result = predictions.copy()
    result["future_close"] = future_close.reindex(result.index)
    result["realized_return"] = result["future_close"] / primary["close"].reindex(result.index) - 1
    result["actual"] = labels.reindex(result.index)
    return result


def predicted_class(probabilities: pd.DataFrame) -> pd.Series:
    """Most likely class (-1/0/1) per row of SELL/HOLD/BUY probabilities."""
    names = probabilities[CLASS_NAMES].astype(float).idxmax(axis=1)
    return names.map({"SELL": -1, "HOLD": 0, "BUY": 1})
