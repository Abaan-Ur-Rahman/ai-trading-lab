"""Technical indicator calculations."""

from __future__ import annotations

import pandas as pd
import pandas_ta_classic as ta


class TechnicalIndicators:
    """Calculate technical indicators on OHLCV market data."""

    @staticmethod
    def _validate_close_column(dataframe: pd.DataFrame) -> None:
        """Ensure the dataframe contains a close column."""
        if "close" not in dataframe.columns:
            raise ValueError("DataFrame must contain a 'close' column")

    def exponential_moving_average(
        self,
        dataframe: pd.DataFrame,
        length: int,
    ) -> pd.Series:
        """Calculate the Exponential Moving Average (EMA)."""

        self._validate_close_column(dataframe)

        return ta.ema(
            dataframe["close"],
            length=length,
        )

    def relative_strength_index(
        self,
        dataframe: pd.DataFrame,
        length: int = 14,
    ) -> pd.Series:
        """Calculate the Relative Strength Index (RSI)."""

        self._validate_close_column(dataframe)

        return ta.rsi(
            dataframe["close"],
            length=length,
        )

    def average_true_range(
        self,
        dataframe: pd.DataFrame,
        length: int = 14,
    ) -> pd.Series:
        """Calculate the Average True Range (ATR)."""

        required_columns = {
            "high",
            "low",
            "close",
        }

        missing = required_columns.difference(dataframe.columns)

        if missing:
            raise ValueError(
                "DataFrame must contain high, low, and close columns",
            )

        return ta.atr(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            length=length,
        )

    def macd(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate MACD."""
        ...