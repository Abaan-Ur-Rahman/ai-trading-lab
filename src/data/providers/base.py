"""Provider interface for market data sources.

This module defines the abstract base class that all market data providers
must implement. Concrete providers (e.g., Binance, Coinbase, mock) should
inherit from this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from data.models import MarketCandle


class MarketDataProvider(ABC):
    """Abstract base class for market data providers.
    
    Subclasses must implement the get_candles() method to fetch OHLCV data
    from their respective data sources. This interface ensures a consistent
    contract across all provider implementations.
    """

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[MarketCandle]:
        """Fetch market candles for a given symbol and timeframe.

        Parameters
        ----------
        symbol : str
            Market symbol or asset identifier (e.g., "BTC-USD").
        timeframe : str
            Candle timeframe (e.g., "1m", "5m", "1h", "1d").
        limit : int, optional
            Maximum number of candles to fetch (default: 100).

        Returns
        -------
        list[MarketCandle]
            A list of MarketCandle objects sorted by timestamp (oldest first).
            Should contain at most `limit` candles.
        """
        ...
