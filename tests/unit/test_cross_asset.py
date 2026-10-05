"""Tests for features.cross_asset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.cross_asset import align_secondary_close, build_cross_asset_features


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")


def _random_walk(n: int, seed: int, start: float) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(0, 0.001, n))), index=_index(n))


def test_align_forward_fills_without_looking_ahead() -> None:
    primary_index = _index(4)
    secondary = pd.Series([1.0, 3.0], index=primary_index[[1, 3]])

    aligned = align_secondary_close(primary_index, secondary)

    assert np.isnan(aligned.iloc[0])
    assert aligned.tolist()[1:] == [1.0, 1.0, 3.0]


def test_output_columns_and_warm_up() -> None:
    primary = _random_walk(60, seed=1, start=2000.0)
    secondary = _random_walk(60, seed=2, start=1.1)

    features = build_cross_asset_features(primary, secondary, corr_window=10)

    assert list(features.columns) == ["secondary_log_return", "ratio_log_return", "rolling_correlation"]
    assert features["rolling_correlation"].iloc[:10].isna().all()
    assert features.iloc[10:].notna().all().all()


def test_flat_secondary_window_gives_zero_correlation_not_nan() -> None:
    n = 60
    primary = _random_walk(n, seed=3, start=2000.0)
    secondary = _random_walk(n, seed=4, start=1.1)
    # Secondary stops moving for 25 bars (e.g. forward-filled through a data gap).
    secondary.iloc[20:45] = secondary.iloc[20]

    features = build_cross_asset_features(primary, secondary, corr_window=10)
    correlation = features["rolling_correlation"]

    # Windows lying entirely inside the flat stretch (returns at 21..44).
    fully_flat = correlation.iloc[30:45]
    assert (fully_flat == 0.0).all()
    assert correlation.iloc[10:].notna().all()


def test_correlation_stays_within_bounds() -> None:
    primary = _random_walk(200, seed=5, start=2000.0)
    secondary = _random_walk(200, seed=6, start=1.1)

    correlation = build_cross_asset_features(primary, secondary, corr_window=20)["rolling_correlation"].dropna()

    assert correlation.between(-1.0, 1.0).all()


def test_identical_series_correlate_perfectly() -> None:
    primary = _random_walk(50, seed=7, start=2000.0)

    correlation = build_cross_asset_features(primary, primary, corr_window=10)["rolling_correlation"]

    assert correlation.iloc[10:].to_numpy() == pytest.approx(1.0)


def test_rejects_too_small_window() -> None:
    series = _random_walk(30, seed=8, start=1.0)
    with pytest.raises(ValueError):
        build_cross_asset_features(series, series, corr_window=1)