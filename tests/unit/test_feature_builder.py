"""Unit tests for the feature builder."""

from __future__ import annotations

import pandas as pd
import pytest

from features.feature_builder import build_features, multi_period_log_return


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


@pytest.fixture
def longer_close_series() -> pd.Series:
    """Return a 25-row close series, long enough for 10- and 20-period returns."""
    return pd.Series(
        [100, 102, 101, 103, 105, 104, 106, 108, 107, 110,
         112, 111, 113, 115, 114, 116, 118, 117, 119, 121,
         120, 122, 124, 123, 125],
        dtype=float,
    )


def test_build_features_returns_expected_columns(
    sample_ohlc_dataframe: pd.DataFrame,
) -> None:
    """build_features should return exactly the eight expected columns."""
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
        "return_3",
        "return_10",
        "return_20",
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
    assert last_row["return_3"] == pytest.approx(0.037041, abs=1e-6)
    assert last_row["ema_gap_pct"] == pytest.approx(0.015291, abs=1e-6)
    assert last_row["rsi"] == pytest.approx(81.917578, abs=1e-6)
    assert last_row["atr_pct"] == pytest.approx(0.027418, abs=1e-6)
    assert last_row["macd_hist_pct"] == pytest.approx(0.000764, abs=1e-6)


def test_build_features_return_10_and_return_20_are_nan_with_insufficient_history(
    sample_ohlc_dataframe: pd.DataFrame,
) -> None:
    """With only 10 rows, return_10 and return_20 cannot be computed for any row."""
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

    assert result["return_10"].isna().all()
    assert result["return_20"].isna().all()


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


# --- multi_period_log_return (return_3 / return_10 / return_20) ---


def test_multi_period_log_return_computes_expected_values(
    longer_close_series: pd.Series,
) -> None:
    """Hand-verified log return values at specific checkpoints."""
    result_10 = multi_period_log_return(longer_close_series, periods=10)
    result_20 = multi_period_log_return(longer_close_series, periods=20)

    assert result_10.iloc[-1] == pytest.approx(0.092115, abs=1e-6)
    assert result_10.iloc[15] == pytest.approx(0.109199, abs=1e-6)
    assert result_20.iloc[-1] == pytest.approx(0.174353, abs=1e-6)


@pytest.mark.parametrize("periods", [3, 10, 20])
def test_multi_period_log_return_produces_nan_for_insufficient_history(
    longer_close_series: pd.Series,
    periods: int,
) -> None:
    """The first `periods` rows have no old-enough price and must be NaN."""
    result = multi_period_log_return(longer_close_series, periods=periods)

    assert result.iloc[:periods].isna().all()
    assert result.iloc[periods:].notna().all()


@pytest.mark.parametrize("periods", [0, -1, -5])
def test_multi_period_log_return_rejects_non_positive_periods(
    longer_close_series: pd.Series,
    periods: int,
) -> None:
    """periods must be a positive integer."""
    with pytest.raises(ValueError, match="positive integer"):
        multi_period_log_return(longer_close_series, periods=periods)


def test_multi_period_log_return_is_zero_for_constant_price() -> None:
    """A flat price series has zero log return everywhere it can be computed."""
    constant_close = pd.Series([100.0] * 10)

    result = multi_period_log_return(constant_close, periods=3)

    assert (result.dropna() == 0.0).all()


def test_multi_period_log_return_uses_only_past_and_current_data(
    longer_close_series: pd.Series,
) -> None:
    """Changing a later price must not change any earlier computed return (no leakage)."""
    perturbed = longer_close_series.copy()
    perturbed.iloc[-1] = 99_999.0

    original_result = multi_period_log_return(longer_close_series, periods=3)
    perturbed_result = multi_period_log_return(perturbed, periods=3)

    # Every row except the last one (the only row reading the perturbed
    # price) must be unaffected.
    pd.testing.assert_series_equal(
        original_result.iloc[:-1],
        perturbed_result.iloc[:-1],
    )


def test_multi_period_log_return_preserves_index() -> None:
    """The result must keep the same index as the input series."""
    custom_index = pd.date_range("2024-01-01", periods=10, freq="h")
    close = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 110], dtype=float)
    close.index = custom_index

    result = multi_period_log_return(close, periods=3)

    pd.testing.assert_index_equal(result.index, custom_index)