"""Trading signal generation framework."""

from __future__ import annotations

from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field


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

    def ema_crossover(
        self,
        dataframe: pd.DataFrame,
    ) -> SignalResult:
        """Generate a signal based on EMA crossover."""
        raise NotImplementedError

    def rsi_signal(
        self,
        dataframe: pd.DataFrame,
    ) -> SignalResult:
        """Generate a signal based on RSI thresholds."""
        raise NotImplementedError

    def macd_signal(
        self,
        dataframe: pd.DataFrame,
    ) -> SignalResult:
        """Generate a signal based on MACD."""
        raise NotImplementedError

    def combined_signal(
        self,
        dataframe: pd.DataFrame,
    ) -> SignalResult:
        """Combine multiple signals into a single confidence-scored signal."""
        raise NotImplementedError