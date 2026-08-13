"""
Utilities for converting market data into pandas DataFrames.
"""

from __future__ import annotations

import pandas as pd

from data.models import MarketCandle


def candles_to_dataframe(
    candles: list[MarketCandle],
) -> pd.DataFrame:
    """Convert a list of MarketCandle objects into a pandas DataFrame."""

    rows = [
        {
            "timestamp": candle.timestamp,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
        }
        for candle in candles
    ]

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    return df