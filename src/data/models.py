"""
Data models for AI Trading Lab.

This module contains small, focused Pydantic models used to represent
market data structures. Keep these models minimal and validation-focused.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator


class MarketCandle(BaseModel):
    """A validated representation of a single OHLCV market candle.

    Attributes
    ----------
    timestamp:
        The timestamp associated with the candle.
    symbol:
        Market symbol/asset, normalized to uppercase.
    timeframe:
        Timeframe identifier such as ``1m``, ``5m``, or ``1h``.
    open:
        Opening price of the candle.
    high:
        Highest price reached during the candle.
    low:
        Lowest price reached during the candle.
    close:
        Closing price of the candle.
    volume:
        Optional traded volume associated with the candle.
    """

    timestamp: datetime
    symbol: str = Field(..., min_length=1)
    timeframe: str = Field(..., min_length=1)

    open: float
    high: float
    low: float
    close: float

    volume: float | None = None

    model_config = {
        "str_strip_whitespace": True,
        "validate_assignment": True,
    }

    @field_validator("symbol", mode="before")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        """Normalize the market symbol to uppercase."""
        if not isinstance(value, str):
            raise TypeError("symbol must be a string")

        return value.strip().upper()

    @field_validator("open", "high", "low", "close", mode="after")
    @classmethod
    def _validate_positive_price(cls, value: float) -> float:
        """Ensure that all OHLC prices are positive."""
        if value <= 0:
            raise ValueError("price fields must be positive")

        return value

    @model_validator(mode="after")
    def _validate_price_relationships(self) -> MarketCandle:
        """Validate logical relationships between OHLC prices."""
        if self.high < self.low:
            raise ValueError(
                "high must be greater than or equal to low"
            )

        if self.high < self.open or self.high < self.close:
            raise ValueError(
                "high must be greater than or equal to open and close"
            )

        if self.low > self.open or self.low > self.close:
            raise ValueError(
                "low must be less than or equal to open and close"
            )

        return self