"""
Market data service.

Provides a high-level interface for retrieving market data while
remaining independent of any specific data provider.
"""

from __future__ import annotations

import re
import time
from datetime import timedelta
from typing import Callable

from data.models import MarketCandle
from data.providers.base import MarketDataProvider

_TIMEFRAME_PATTERN = re.compile(r"^(\d+)(min|h|day|week)$")
_UNIT_TO_TIMEDELTA_KWARGS = {"min": "minutes", "h": "hours", "day": "days", "week": "weeks"}


def _parse_timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Convert a Twelve Data interval string (e.g. '1h', '15min') to a timedelta.

    Supports minute/hour/day/week intervals. Month-based intervals aren't
    supported since a calendar month isn't a fixed duration.

    Raises:
        ValueError: If `timeframe` doesn't match a supported pattern.
    """
    match = _TIMEFRAME_PATTERN.match(timeframe)
    if not match:
        raise ValueError(
            f"Unsupported timeframe for pagination: {timeframe!r}. "
            "Supported patterns: <N>min, <N>h, <N>day, <N>week (e.g. '1h', '15min').",
        )
    quantity, unit = match.groups()
    return timedelta(**{_UNIT_TO_TIMEDELTA_KWARGS[unit]: int(quantity)})


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

    def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        total_candles: int,
        page_size: int = 5000,
        sleep_seconds: float = 8.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> list[MarketCandle]:
        """Fetch more candles than a single request can return.

        Pages backward in time: each request asks the provider for
        `page_size` candles ending just before the earliest candle
        already fetched, repeating until `total_candles` is reached or
        the provider returns fewer candles than requested (meaning the
        start of available history has been reached).

        Sleeps `sleep_seconds` between requests (not after the last one)
        to stay within the provider's rate limit. `sleep_fn` is
        injectable so tests don't actually wait.

        Raises:
            ValueError: If total_candles or page_size is not positive, or
                if `timeframe` isn't a supported interval string.
        """
        if total_candles <= 0:
            raise ValueError("total_candles must be a positive integer")
        if page_size <= 0:
            raise ValueError("page_size must be a positive integer")

        step = _parse_timeframe_to_timedelta(timeframe)

        all_candles: list[MarketCandle] = []
        end_date: str | None = None
        is_first_request = True

        while len(all_candles) < total_candles:
            if not is_first_request:
                sleep_fn(sleep_seconds)
            is_first_request = False

            remaining = total_candles - len(all_candles)
            page_limit = min(page_size, remaining)

            page = self._provider.get_candles(
                symbol=symbol,
                timeframe=timeframe,
                limit=page_limit,
                end_date=end_date,
            )

            if not page:
                break

            all_candles = page + all_candles

            earliest_timestamp = page[0].timestamp
            end_date = (earliest_timestamp - step).strftime("%Y-%m-%d %H:%M:%S")

            if len(page) < page_limit:
                break

        return all_candles