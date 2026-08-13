"""Technical indicator calculations."""

from __future__ import annotations

import pandas as pd


class TechnicalIndicators:
    """Calculate technical indicators on OHLCV market data."""

    def exponential_moving_average(
        self,
        dataframe: pd.DataFrame,
        length: int,
    ) -> pd.Series:
        """Calculate EMA."""
        ...

    def relative_strength_index(
        self,
        dataframe: pd.DataFrame,
        length: int = 14,
    ) -> pd.Series:
        """Calculate RSI."""
        ...

    def average_true_range(
        self,
        dataframe: pd.DataFrame,
        length: int = 14,
    ) -> pd.Series:
        """Calculate ATR."""
        ...

    def macd(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate MACD."""
        ...