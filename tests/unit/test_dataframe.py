from datetime import datetime, timezone

from data.dataframe import candles_to_dataframe
from data.models import MarketCandle


def test_dataframe_conversion():
    candles = [
        MarketCandle(
            timestamp=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
            symbol="BTC/USD",
            timeframe="1h",
            open=100,
            high=105,
            low=95,
            close=103,
            volume=1000,
        ),
        MarketCandle(
            timestamp=datetime(2024, 1, 1, 13, tzinfo=timezone.utc),
            symbol="BTC/USD",
            timeframe="1h",
            open=103,
            high=108,
            low=101,
            close=106,
            volume=1100,
        ),
    ]

    df = candles_to_dataframe(candles)

    assert len(df) == 2
    assert list(df.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert df.index.name == "timestamp"