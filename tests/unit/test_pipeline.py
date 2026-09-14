"""Unit tests for the feature-engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.pipeline import build_feature_dataset


def _make_ohlc_dataframe(rows: int = 20, seed: int = 42) -> pd.DataFrame:
    """Build a reproducible synthetic OHLC dataframe for pipeline tests."""
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.standard_normal(rows)), dtype=float)

    return pd.DataFrame(
        {
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )


_SMALL_PERIODS = {
    "ema_fast": 3,
    "ema_slow": 6,
    "rsi_length": 6,
    "atr_length": 3,
    "macd_fast": 3,
    "macd_slow": 6,
    "macd_signal": 2,
    "horizon": 2,
    "threshold": 0.02,
}


def test_build_feature_dataset_has_no_nans() -> None:
    """The final dataset must contain zero NaN values anywhere."""
    dataframe = _make_ohlc_dataframe()

    dataset = build_feature_dataset(dataframe, min_rows=1, **_SMALL_PERIODS)

    assert dataset.isna().sum().sum() == 0
    assert len(dataset) > 0


def test_build_feature_dataset_raises_below_min_rows() -> None:
    """Too few usable rows after dropna should raise ValueError."""
    dataframe = _make_ohlc_dataframe(rows=20)

    with pytest.raises(ValueError):
        build_feature_dataset(dataframe, min_rows=100, **_SMALL_PERIODS)


def test_build_feature_dataset_rejects_non_positive_min_rows() -> None:
    """min_rows must be greater than 0."""
    dataframe = _make_ohlc_dataframe()

    with pytest.raises(ValueError):
        build_feature_dataset(dataframe, min_rows=0, **_SMALL_PERIODS)


def test_build_feature_dataset_does_not_mutate_input() -> None:
    """build_feature_dataset must not modify the caller's original dataframe."""
    dataframe = _make_ohlc_dataframe()
    original = dataframe.copy(deep=True)

    build_feature_dataset(dataframe, min_rows=1, **_SMALL_PERIODS)

    pd.testing.assert_frame_equal(dataframe, original)


def test_build_feature_dataset_prevents_look_ahead_bias() -> None:
    """Mutating future OHLC rows must not change earlier feature/label rows.

    This is the leakage test: row 10 sits well past every indicator's
    warm-up window (trailing-only, so unaffected by future data by
    construction) and well before row 19 within the label horizon
    (horizon=2, so row 10's label only reads row 12 — nowhere near
    row 19). If row 10 changes after mutating row 19, something in the
    pipeline is leaking future information into the past.
    """
    dataframe = _make_ohlc_dataframe(rows=20)

    baseline = build_feature_dataset(dataframe, min_rows=1, **_SMALL_PERIODS)

    mutated = dataframe.copy(deep=True)
    mutated.loc[19, ["close", "high", "low"]] = [99999.0, 100000.0, 99998.0]

    result = build_feature_dataset(mutated, min_rows=1, **_SMALL_PERIODS)

    assert 10 in baseline.index
    assert 10 in result.index
    pd.testing.assert_series_equal(baseline.loc[10], result.loc[10])