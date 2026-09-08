"""Trading signal generation framework."""

from __future__ import annotations

from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

from features.technical_indicators import TechnicalIndicators


class Signal(str, Enum):
    """Possible trading signal directions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalResult(BaseModel):
    """Result of a single signal calculation."""

    direction: Signal
    confidence: float = Field(ge=0.0, le=1.0)


class SignalGenerator:
    """Generate trading signals from technical indicators."""

    def __init__(self, indicators: TechnicalIndicators | None = None) -> None:
        self._indicators = indicators or TechnicalIndicators()

    def ema_crossover(
        self,
        dataframe: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
    ) -> SignalResult:
        """Generate a signal based on EMA crossover.

        Confidence is binary at this stage (1.0 for a confirmed cross,
        0.0 for no cross). Step 7 will replace this with a proper score.
        """
        if fast >= slow:
            raise ValueError("fast length must be less than slow length")

        min_required_rows = slow + 1
        if len(dataframe) < min_required_rows:
            raise ValueError(
                f"DataFrame must contain at least {min_required_rows} rows "
                f"for EMA crossover with slow={slow}",
            )

        fast_ema = self._indicators.exponential_moving_average(dataframe, length=fast)
        slow_ema = self._indicators.exponential_moving_average(dataframe, length=slow)

        prev_fast, curr_fast = fast_ema.iloc[-2], fast_ema.iloc[-1]
        prev_slow, curr_slow = slow_ema.iloc[-2], slow_ema.iloc[-1]

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return SignalResult(direction=Signal.BUY, confidence=1.0)

        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return SignalResult(direction=Signal.SELL, confidence=1.0)

        return SignalResult(direction=Signal.HOLD, confidence=0.0)

    def rsi_signal(
        self,
        dataframe: pd.DataFrame,
        length: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> SignalResult:
        """Generate a signal based on the latest RSI value.

        Confidence is binary at this stage (1.0 for a threshold breach,
        0.0 for no breach). Step 7 will replace this with a proper score.
        """
        if oversold >= overbought:
            raise ValueError("oversold threshold must be less than overbought threshold")

        rsi = self._indicators.relative_strength_index(dataframe, length=length)
        latest_rsi = rsi.iloc[-1]

        if latest_rsi <= oversold:
            return SignalResult(direction=Signal.BUY, confidence=1.0)

        if latest_rsi >= overbought:
            return SignalResult(direction=Signal.SELL, confidence=1.0)

        return SignalResult(direction=Signal.HOLD, confidence=0.0)

    def macd_signal(
        self,
        dataframe: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> SignalResult:
        """Generate a signal based on the MACD line crossing its signal line.

        Confidence is binary at this stage (1.0 for a confirmed cross,
        0.0 for no cross). Step 7 will replace this with a proper score.
        """
        macd_df = self._indicators.macd(dataframe, fast=fast, slow=slow, signal=signal)

        macd_line = macd_df[f"MACD_{fast}_{slow}_{signal}"]
        signal_line = macd_df[f"MACDs_{fast}_{slow}_{signal}"]

        prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
        prev_signal, curr_signal = signal_line.iloc[-2], signal_line.iloc[-1]

        if pd.isna(prev_macd) or pd.isna(curr_macd) or pd.isna(prev_signal) or pd.isna(curr_signal):
            raise ValueError("Not enough data to detect a MACD crossover")

        if prev_macd <= prev_signal and curr_macd > curr_signal:
            return SignalResult(direction=Signal.BUY, confidence=1.0)

        if prev_macd >= prev_signal and curr_macd < curr_signal:
            return SignalResult(direction=Signal.SELL, confidence=1.0)

        return SignalResult(direction=Signal.HOLD, confidence=0.0)
    def combined_signal(
        self,
        dataframe: pd.DataFrame,
    ) -> SignalResult:
        """Combine multiple signals into a single confidence-scored signal."""
        raise NotImplementedError