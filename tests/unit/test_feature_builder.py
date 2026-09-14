"""Unit tests for the feature builder."""

from __future__ import annotations

import pandas as pd
import pytest

from features.feature_builder import build_features


@pytest.fixture
def sample_ohlc_dataframe() -> pd.DataFrame:
    """Return a small OHLC dataframe for feature-building tests."""
    close = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 110], dtype=float)

    return pd.DataFrame(
        {
            "high": close + 1,
            "low": close - 1,
            "close": close,
        }
    )


def test_build_features_returns_expected_columns(
    sample_ohlc_dataframe: pd.DataFrame,
) -> None:
    """build_features should return exactly the five expected columns."""
    result = build_features(
        sample_ohlc_dataframe,
        ema_fast=3,
        ema_slow=6,
        rsi_length=6,
        atr_length=3,
        macd_fast=3,
        macd_slow=6,
        macd_signal=2,
    )

    assert set(result.columns) == {
        "log_return",
        "ema_gap_pct",
        "rsi",
        "atr_pct",
        "macd_hist_pct",
    }


def test_build_features_preserves_row_count(
    sample_ohlc_dataframe: pd.DataFrame,
) -> None:
    """build_features should not drop any rows itself."""
    result = build_features(
        sample_ohlc_dataframe,
        ema_fast=3,
        ema_slow=6,
        rsi_length=6,
        atr_length=3,
        macd_fast=3,
        macd_slow=6,
        macd_signal=2,
    )

    assert len(result) == len(sample_ohlc_dataframe)


def test_build_features_requires_ohlc_columns() -> None:
    """build_features should require high, low, and close columns."""
    dataframe = pd.DataFrame({"close": [100, 101, 102]})

    with pytest.raises(ValueError):
        build_features(dataframe)


def test_build_features_computes_expected_values(
    sample_ohlc_dataframe: pd.DataFrame,
) -> None:
    """Feature values at the last row should match hand-verified expected values."""
    result = build_features(
        sample_ohlc_dataframe,
        ema_fast=3,
        ema_slow=6,
        rsi_length=6,
        atr_length=3,
        macd_fast=3,
        macd_slow=6,
        macd_signal=2,
    )

    last_row = result.iloc[-1]

    assert last_row["log_return"] == pytest.approx(0.027652, abs=1e-6)
    assert last_row["ema_gap_pct"] == pytest.approx(0.015291, abs=1e-6)
    assert last_row["rsi"] == pytest.approx(81.917578, abs=1e-6)
    assert last_row["atr_pct"] == pytest.approx(0.027418, abs=1e-6)
    assert last_row["macd_hist_pct"] == pytest.approx(0.000764, abs=1e-6)


def test_build_features_does_not_mutate_input(
    sample_ohlc_dataframe: pd.DataFrame,
) -> None:
    """build_features must not modify the caller's original dataframe."""
    original = sample_ohlc_dataframe.copy(deep=True)

    build_features(
        sample_ohlc_dataframe,
        ema_fast=3,
        ema_slow=6,
        rsi_length=6,
        atr_length=3,
        macd_fast=3,
        macd_slow=6,
        macd_signal=2,
    )

    pd.testing.assert_frame_equal(sample_ohlc_dataframe, original)