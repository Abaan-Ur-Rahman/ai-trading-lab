"""Tests for forward.tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forward.tracking import append_bars, attach_outcomes, bars_ready_to_predict, complete_new_bars, predicted_class


def _bars(start: str, n: int, closes: list[float] | None = None) -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="h", tz="UTC")
    close = closes if closes is not None else list(range(100, 100 + n))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=index)


def test_complete_new_bars_drops_still_forming_and_already_stored_bars() -> None:
    fetched = _bars("2026-10-02 10:00", 5)  # 10:00 .. 14:00
    now = pd.Timestamp("2026-10-02 14:30", tz="UTC")  # the 14:00 bar is still forming

    new = complete_new_bars(fetched, after=pd.Timestamp("2026-10-02 10:00", tz="UTC"), now=now)

    assert [t.hour for t in new.index] == [11, 12, 13]


def test_bar_counts_as_complete_exactly_when_its_hour_ends() -> None:
    fetched = _bars("2026-10-02 13:00", 1)

    new = complete_new_bars(fetched, after=pd.Timestamp("2026-10-01", tz="UTC"),
                            now=pd.Timestamp("2026-10-02 14:00", tz="UTC"))

    assert len(new) == 1


def test_append_bars_never_overwrites_stored_bars() -> None:
    stored = _bars("2026-10-02 10:00", 2, closes=[1.0, 2.0])
    refetched = _bars("2026-10-02 11:00", 2, closes=[999.0, 3.0])  # 11:00 revised by the provider

    merged = append_bars(stored, refetched)

    assert merged["close"].tolist() == [1.0, 2.0, 3.0]


def test_append_bars_to_empty_store() -> None:
    new = _bars("2026-10-02 10:00", 3)

    pd.testing.assert_frame_equal(append_bars(None, new), new)


def test_bars_ready_to_predict_waits_for_every_secondary_and_skips_logged_bars() -> None:
    primary = pd.date_range("2026-10-02 00:00", periods=6, freq="h", tz="UTC")  # 00:00 .. 05:00
    logged = primary[[1]]

    ready = bars_ready_to_predict(
        primary,
        {"EURUSD": primary[5], "USDJPY": primary[3]},  # USD/JPY only up to 03:00 so far
        logged,
        start_after=primary[0],
    )

    assert [t.hour for t in ready] == [2, 3]


def test_attach_outcomes_uses_training_label_definition_and_waits_for_horizon() -> None:
    primary = _bars("2026-10-02 00:00", 8, closes=[100, 100, 100, 101, 100, 100, 100, 100])
    predictions = pd.DataFrame({"SELL": 0.1, "HOLD": 0.2, "BUY": 0.7}, index=primary.index[:4])

    scored = attach_outcomes(predictions, primary, horizon=2, threshold=0.005)

    # Bar 1: close 100 -> 101 two bars later = +1% > 0.5% => BUY (1).
    assert scored["actual"].iloc[1] == 1
    assert scored["realized_return"].iloc[1] == pytest.approx(0.01)
    # Bar 0: 100 -> 100 => HOLD.
    assert scored["actual"].iloc[0] == 0


def test_attach_outcomes_leaves_recent_bars_pending() -> None:
    primary = _bars("2026-10-02 00:00", 4)
    predictions = pd.DataFrame({"SELL": 0.1, "HOLD": 0.2, "BUY": 0.7}, index=primary.index)

    scored = attach_outcomes(predictions, primary, horizon=2, threshold=0.005)

    assert scored["actual"].iloc[-2:].isna().all()
    assert scored["actual"].iloc[:2].notna().all()


def test_predicted_class_takes_most_likely() -> None:
    proba = pd.DataFrame({"SELL": [0.6, 0.1, 0.2], "HOLD": [0.3, 0.8, 0.2], "BUY": [0.1, 0.1, 0.6]})

    assert predicted_class(proba).tolist() == [-1, 0, 1]
    assert np.issubdtype(predicted_class(proba).dtype, np.integer)
