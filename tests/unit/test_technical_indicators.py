"""Unit tests for TechnicalIndicators."""

from __future__ import annotations

import pandas as pd

from features.technical_indicators import TechnicalIndicators


def sample_dataframe() -> pd.DataFrame:
    """Return a small OHLC dataframe."""

    return pd.DataFrame(
        {
            "open": [100,101,102,103,104,105,106,107,108,109],
            "high": [101,102,103,104,105,106,107,108,109,110],
            "low": [99,100,101,102,103,104,105,106,107,108],
            "close": [100,101,102,103,104,105,106,107,108,109],
            "volume": [10]*10,
        }
    )


def test_indicator_class_exists():
    indicator = TechnicalIndicators()
    assert indicator is not None


def test_dataframe_fixture():
    df = sample_dataframe()

    assert not df.empty
    assert len(df) == 10

def test_ema_returns_series():
    df = sample_dataframe()

    indicator = TechnicalIndicators()

    ema = indicator.exponential_moving_average(
        df,
        length=3,
    )

    assert isinstance(ema, pd.Series)

    assert len(ema) == len(df)