"""Twelve Data market data provider for AI Trading Lab.

This module provides access to OHLCV market data through the
Twelve Data REST API.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv

from data.models import MarketCandle
from data.providers.base import MarketDataProvider


class TwelveDataProvider(MarketDataProvider):
    """Market data provider backed by the Twelve Data REST API."""

    BASE_URL = "https://api.twelvedata.com"
    ENDPOINT = "/time_series"
    TIMEOUT = 10.0

    def __init__(self) -> None:
        """Initialize the Twelve Data provider.

        Raises
        ------
        ValueError
            If the Twelve Data API key is not configured.
        """
        load_dotenv()

        api_key = os.getenv("TWELVE_DATA_API_KEY")

        if not api_key:
            raise ValueError(
                "TWELVE_DATA_API_KEY environment variable is not configured"
            )

        self._api_key = api_key

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[MarketCandle]:
        """Fetch OHLCV candles from Twelve Data.

        Parameters
        ----------
        symbol:
            Market symbol, such as ``XAU/USD`` or ``AAPL``.
        timeframe:
            Candle interval, such as ``1h`` or ``1day``.
        limit:
            Maximum number of candles to retrieve.

        Returns
        -------
        list[MarketCandle]
            Validated candles sorted from oldest to newest.

        Raises
        ------
        ValueError
            If ``limit`` is not a positive integer.
        RuntimeError
            If Twelve Data returns an API-level error or malformed data.
        httpx.HTTPError
            If the HTTP request itself fails.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be a positive integer")

        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        params = {
            "symbol": symbol,
            "interval": timeframe,
            "outputsize": limit,
            "timezone": "UTC",
        }

        headers = {
            "Authorization": f"apikey {self._api_key}",
        }

        url = f"{self.BASE_URL}{self.ENDPOINT}"

        try:
            response = httpx.get(
                url,
                params=params,
                headers=headers,
                timeout=self.TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            raise

        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Twelve Data returned an invalid JSON response"
            ) from exc

        if data.get("status") == "error":
            message = data.get("message", "Unknown Twelve Data API error")
            raise RuntimeError(f"Twelve Data API error: {message}")

        values = data.get("values")

        if not isinstance(values, list):
            raise RuntimeError(
                "Unexpected Twelve Data response: 'values' must be a list"
            )

        candles = [
            self._record_to_candle(
                symbol=symbol,
                timeframe=timeframe,
                record=record,
            )
            for record in values
        ]

        candles.sort(key=lambda candle: candle.timestamp)

        return candles

    @staticmethod
    def _record_to_candle(
        symbol: str,
        timeframe: str,
        record: dict[str, Any],
    ) -> MarketCandle:
        """Convert one Twelve Data record into a MarketCandle."""
        try:
            datetime_value = record["datetime"]

            open_price = float(record["open"])
            high_price = float(record["high"])
            low_price = float(record["low"])
            close_price = float(record["close"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Invalid OHLC data received from Twelve Data"
            ) from exc

        if not isinstance(datetime_value, str):
            raise RuntimeError(
                "Invalid datetime received from Twelve Data"
            )

        try:
            timestamp = datetime.strptime(
                datetime_value,
                "%Y-%m-%d %H:%M:%S",
            ).replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid datetime format received from Twelve Data: "
                f"{datetime_value}"
            ) from exc

        volume: float | None = None

        raw_volume = record.get("volume")

        if raw_volume is not None and raw_volume != "":
            try:
                volume = float(raw_volume)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Invalid volume data received from Twelve Data"
                ) from exc

        return MarketCandle(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )