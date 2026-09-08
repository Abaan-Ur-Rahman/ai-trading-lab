"""Technical indicator calculations."""

from __future__ import annotations

import pandas as pd
import pandas_ta_classic as ta


class TechnicalIndicators:
    """Calculate technical indicators on OHLCV market data."""

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        required: set[str],
    ) -> None:
        """Ensure the dataframe contains the required columns."""
        missing = required.difference(dataframe.columns)

        if missing:
            raise ValueError(
                f"DataFrame must contain columns: {sorted(missing)}",
            )

    def exponential_moving_average(
        self,
        dataframe: pd.DataFrame,
        length: int,
    ) -> pd.Series:
        """Calculate the Exponential Moving Average (EMA)."""

        self._validate_columns(dataframe, {"close"})

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

        self._validate_columns(dataframe, {"close"})

        result = ta.rsi(
            dataframe["close"],
            length=length,
        )

        if result is None:
            raise ValueError("Failed to calculate RSI")

        return result
    
    def average_true_range(
        self,
        dataframe: pd.DataFrame,
        length: int = 14,
    ) -> pd.Series:
        """Calculate the Average True Range (ATR)."""

        self._validate_columns(dataframe, {"high", "low", "close"})

        return ta.atr(
            high=dataframe["high"],
            low=dataframe["low"],
            close=dataframe["close"],
            length=length,
        )

    def macd(
        self,
        dataframe: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Calculate the Moving Average Convergence Divergence (MACD)."""

        self._validate_columns(dataframe, {"close"})

        result = ta.macd(
            dataframe["close"],
            fast=fast,
            slow=slow,
            signal=signal,
        )

        if result is None:
            raise ValueError("Failed to calculate MACD")

        return result