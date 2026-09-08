"""Unit tests for TechnicalIndicators."""

from __future__ import annotations

import pandas as pd
import pytest

from features.technical_indicators import TechnicalIndicators


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a small OHLCV dataframe."""

    return pd.DataFrame(
        {
            "open": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
            ],
            "high": [
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
            ],
            "low": [
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
            ],
            "close": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
            ],
            "volume": [
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            ],
        }
    )


def test_indicator_class_exists() -> None:
    """Indicator class can be instantiated."""
    indicator = TechnicalIndicators()

    assert indicator is not None


def test_dataframe_fixture(sample_dataframe: pd.DataFrame) -> None:
    """Fixture returns a valid dataframe."""
    assert not sample_dataframe.empty
    assert len(sample_dataframe) == 10


def test_ema_requires_close_column() -> None:
    """EMA should require a close column."""
    indicator = TechnicalIndicators()

    dataframe = pd.DataFrame(
        {
            "price": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        indicator.exponential_moving_average(
            dataframe,
            length=3,
        )


def test_ema_returns_series(sample_dataframe: pd.DataFrame) -> None:
    """EMA should return a pandas Series."""
    indicator = TechnicalIndicators()

    ema = indicator.exponential_moving_average(
        sample_dataframe,
        length=3,
    )

    assert isinstance(ema, pd.Series)
    assert len(ema) == len(sample_dataframe)

def test_rsi_returns_series(
    sample_dataframe: pd.DataFrame,
) -> None:
    """RSI should return a pandas Series."""

    indicator = TechnicalIndicators()

    rsi = indicator.relative_strength_index(
        sample_dataframe,
        length=3,
    )

    assert isinstance(rsi, pd.Series)

    assert len(rsi) == len(sample_dataframe)

def test_rsi_requires_close_column() -> None:
    """RSI should require a close column."""

    indicator = TechnicalIndicators()

    dataframe = pd.DataFrame(
        {
            "price": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        indicator.relative_strength_index(
            dataframe,
            length=3,
        )

def test_atr_returns_series(
    sample_dataframe: pd.DataFrame,
) -> None:
    """ATR should return a pandas Series."""

    indicator = TechnicalIndicators()

    atr = indicator.average_true_range(
        sample_dataframe,
        length=3,
    )

    assert isinstance(atr, pd.Series)
    assert len(atr) == len(sample_dataframe)

def test_atr_requires_high_low_close_columns() -> None:
    """ATR should require high, low and close columns."""

    indicator = TechnicalIndicators()

    dataframe = pd.DataFrame(
        {
            "close": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        indicator.average_true_range(
            dataframe,
            length=3,
        )

def test_macd_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    """MACD should return a pandas DataFrame."""

    indicator = TechnicalIndicators()

    macd = indicator.macd(
        sample_dataframe,
        fast=3,
        slow=6,
        signal=2,
    )

    assert isinstance(macd, pd.DataFrame)
    assert len(macd) == len(sample_dataframe)

def test_macd_requires_close_column() -> None:
    """MACD should require a close column."""

    indicator = TechnicalIndicators()

    dataframe = pd.DataFrame(
        {
            "price": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        indicator.macd(dataframe)

def test_macd_raises_when_result_is_none(
    sample_dataframe: pd.DataFrame,
) -> None:
    """MACD should raise ValueError when pandas_ta_classic cannot compute a result."""

    indicator = TechnicalIndicators()

    with pytest.raises(ValueError):
        indicator.macd(sample_dataframe)

def test_rsi_raises_when_result_is_none(
    sample_dataframe: pd.DataFrame,
) -> None:
    """RSI should raise ValueError when there isn't enough data."""

    indicator = TechnicalIndicators()

    with pytest.raises(ValueError):
        indicator.relative_strength_index(sample_dataframe)