"""
Market data service.

Provides a high-level interface for retrieving market data while
remaining independent of any specific data provider.
"""

from __future__ import annotations

from data.models import MarketCandle
from data.providers.base import MarketDataProvider


class MarketDataService:
    """High-level service for retrieving market data."""

    def __init__(self, provider: MarketDataProvider) -> None:
        self._provider = provider

    def get_recent_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[MarketCandle]:
        """Return recent market candles."""
        return self._provider.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )