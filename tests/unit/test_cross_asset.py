"""Tests for features.cross_asset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.cross_asset import (
    CROSS_ASSET_COLUMNS,
    align_secondary_close,
    build_cross_asset_features,
    build_prefixed_cross_asset_features,
    cross_asset_column_names,
    synthetic_dollar_index,
)


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


def test_dollar_index_rises_when_dollar_strengthens_against_both() -> None:
    idx = _index(3)
    eurusd = pd.Series([1.10, 1.09, 1.08], index=idx)  # dollar up vs EUR
    usdjpy = pd.Series([150.0, 151.0, 152.0], index=idx)  # dollar up vs JPY

    index = synthetic_dollar_index(eurusd, usdjpy)

    assert index.is_monotonic_increasing
    assert index.name == "close"


def test_dollar_index_is_equal_weighted_geometric_mean_of_dollar_sides() -> None:
    idx = _index(2)
    eurusd = pd.Series([1.0, 1.0], index=idx)
    usdjpy = pd.Series([100.0, 121.0], index=idx)

    index = synthetic_dollar_index(eurusd, usdjpy)

    # Only USD/JPY moved, by +21%; equal weighting passes on sqrt(1.21) = 1.1.
    assert index.iloc[1] / index.iloc[0] == pytest.approx(1.1)


def test_dollar_index_aligns_mismatched_timestamps_causally() -> None:
    idx = _index(4)
    eurusd = pd.Series([1.0, 1.0, 1.0], index=idx[[0, 1, 3]])
    usdjpy = pd.Series([100.0, 144.0], index=idx[[1, 2]])

    index = synthetic_dollar_index(eurusd, usdjpy)

    # idx[0] has no USD/JPY yet -> dropped; idx[3] carries USD/JPY's last value.
    assert list(index.index) == list(idx[1:])
    assert index.loc[idx[3]] == pytest.approx(index.loc[idx[2]])
    assert index.loc[idx[2]] / index.loc[idx[1]] == pytest.approx(1.2)


def test_prefixed_features_match_individual_builds_with_prefixes() -> None:
    primary = _random_walk(60, seed=11, start=2000.0)
    eurusd = _random_walk(60, seed=12, start=1.1)
    usdjpy = _random_walk(60, seed=13, start=150.0)

    combined = build_prefixed_cross_asset_features(primary, {"EURUSD": eurusd, "USDJPY": usdjpy}, corr_window=10)

    assert list(combined.columns) == cross_asset_column_names(["EURUSD", "USDJPY"])
    assert list(combined.columns)[:3] == [f"eurusd_{c}" for c in CROSS_ASSET_COLUMNS]
    expected = build_cross_asset_features(primary, usdjpy, corr_window=10).add_prefix("usdjpy_")
    pd.testing.assert_frame_equal(combined[expected.columns], expected)


def test_prefixed_features_require_a_secondary() -> None:
    with pytest.raises(ValueError):
        build_prefixed_cross_asset_features(_random_walk(30, seed=14, start=1.0), {})
