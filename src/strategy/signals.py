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

    @staticmethod
    def _scale_confidence(value: float, cap: float) -> float:
        """Scale a non-negative magnitude to [0, 1], capping at `cap`.

        `cap` represents the magnitude treated as "maximum confidence".
        This is a placeholder calibration until it can be tuned against
        real backtest results.
        """
        if cap <= 0:
            return 0.0
        return min(abs(value) / cap, 1.0)

    def ema_crossover(
        self,
        dataframe: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        max_gap_pct: float = 0.02,
    ) -> SignalResult:
        """Generate a signal based on EMA crossover.

        Confidence reflects how wide the gap between the fast and slow EMA
        is, as a percentage of price, relative to `max_gap_pct` (the gap
        size at which confidence saturates at 1.0).
        """
        if fast >= slow:
            raise ValueError("fast length must be less than slow length")

        if max_gap_pct <= 0:
            raise ValueError("max_gap_pct must be greater than 0")

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

        if curr_slow <= 0:
            raise ValueError("slow EMA must be positive to compute confidence")

        gap_pct = abs(curr_fast - curr_slow) / curr_slow
        confidence = min(gap_pct / max_gap_pct, 1.0)

        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return SignalResult(direction=Signal.BUY, confidence=confidence)

        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return SignalResult(direction=Signal.SELL, confidence=confidence)

        return SignalResult(direction=Signal.HOLD, confidence=0.0)

    def rsi_signal(
        self,
        dataframe: pd.DataFrame,
        length: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> SignalResult:
        """Generate a signal based on the latest RSI value.

        Confidence reflects how far past the threshold RSI has moved,
        normalized to the room available on that side of the 0-100 scale.
        """
        if oversold >= overbought:
            raise ValueError("oversold threshold must be less than overbought threshold")

        if not (0 < oversold < 100):
            raise ValueError("oversold must be strictly between 0 and 100")

        if not (0 < overbought < 100):
            raise ValueError("overbought must be strictly between 0 and 100")

        rsi = self._indicators.relative_strength_index(dataframe, length=length)
        latest_rsi = rsi.iloc[-1]

        if pd.isna(latest_rsi):
            raise ValueError("Not enough data to compute a valid RSI value")

        if latest_rsi <= oversold:
            confidence = min((oversold - latest_rsi) / oversold, 1.0)
            return SignalResult(direction=Signal.BUY, confidence=confidence)

        if latest_rsi >= overbought:
            confidence = min((latest_rsi - overbought) / (100 - overbought), 1.0)
            return SignalResult(direction=Signal.SELL, confidence=confidence)

        return SignalResult(direction=Signal.HOLD, confidence=0.0)

    def macd_signal(
        self,
        dataframe: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        max_histogram_pct: float = 0.01,
    ) -> SignalResult:
        """Generate a signal based on the MACD line crossing its signal line.

        Confidence reflects how wide the MACD histogram is, as a
        percentage of price, relative to `max_histogram_pct`.
        """
        if max_histogram_pct <= 0:
            raise ValueError("max_histogram_pct must be greater than 0")

        macd_df = self._indicators.macd(dataframe, fast=fast, slow=slow, signal=signal)

        macd_line = macd_df[f"MACD_{fast}_{slow}_{signal}"]
        signal_line = macd_df[f"MACDs_{fast}_{slow}_{signal}"]

        prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
        prev_signal, curr_signal = signal_line.iloc[-2], signal_line.iloc[-1]

        if pd.isna(prev_macd) or pd.isna(curr_macd) or pd.isna(prev_signal) or pd.isna(curr_signal):
            raise ValueError("Not enough data to detect a MACD crossover")

        latest_close = dataframe["close"].iloc[-1]

        if latest_close <= 0:
            raise ValueError("latest close must be positive to compute confidence")

        histogram_pct = abs(curr_macd - curr_signal) / latest_close
        confidence = min(histogram_pct / max_histogram_pct, 1.0)

        if prev_macd <= prev_signal and curr_macd > curr_signal:
            return SignalResult(direction=Signal.BUY, confidence=confidence)

        if prev_macd >= prev_signal and curr_macd < curr_signal:
            return SignalResult(direction=Signal.SELL, confidence=confidence)

        return SignalResult(direction=Signal.HOLD, confidence=0.0)

    def combined_signal(
        self,
        dataframe: pd.DataFrame,
        ema_fast: int = 12,
        ema_slow: int = 26,
        rsi_length: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_length: int = 9,
    ) -> SignalResult:
        """Combine EMA crossover, RSI, and MACD signals via majority vote.

        Confidence blends breadth (how many signals agree) with depth
        (how strongly the agreeing signals feel about it): the average
        confidence of the signals that voted for the winning direction,
        scaled by the fraction of signals that agree.
        """
        ema_result = self.ema_crossover(dataframe, fast=ema_fast, slow=ema_slow)
        rsi_result = self.rsi_signal(
            dataframe,
            length=rsi_length,
            oversold=rsi_oversold,
            overbought=rsi_overbought,
        )
        macd_result = self.macd_signal(
            dataframe,
            fast=macd_fast,
            slow=macd_slow,
            signal=macd_signal_length,
        )

        results = [ema_result, rsi_result, macd_result]
        directions = [result.direction for result in results]

        vote_counts = {
            direction: directions.count(direction) for direction in set(directions)
        }
        winning_direction, winning_votes = max(vote_counts.items(), key=lambda item: item[1])

        if winning_votes < 2:
            return SignalResult(direction=Signal.HOLD, confidence=0.0)

        winning_confidences = [
            result.confidence for result in results if result.direction == winning_direction
        ]
        average_confidence = sum(winning_confidences) / len(winning_confidences)

        return SignalResult(
            direction=winning_direction,
            confidence=average_confidence * (winning_votes / len(directions)),
        )