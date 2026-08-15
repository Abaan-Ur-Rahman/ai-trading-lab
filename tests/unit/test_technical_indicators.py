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