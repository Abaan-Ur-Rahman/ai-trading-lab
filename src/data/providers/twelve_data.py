"""Twelve Data market data provider for AI Trading Lab.

This module provides access to OHLCV market data through the
Twelve Data REST API.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from data.models import MarketCandle
from data.providers.base import MarketDataProvider
from utils.logger import get_logger

logger = get_logger(__name__)

# Deliberately duplicated from data.services.market_data's own timeframe
# parsing (not imported from there) to avoid a provider depending on a
# service-layer module -- this provider only needs it internally, to
# compute how far back to page when topping up a validation-skip
# shortfall (see get_candles).
_TIMEFRAME_PATTERN = re.compile(r"^(\d+)(min|h|day|week)$")
_UNIT_TO_TIMEDELTA_KWARGS = {"min": "minutes", "h": "hours", "day": "days", "week": "weeks"}


def _timeframe_to_timedelta(timeframe: str) -> timedelta:
    match = _TIMEFRAME_PATTERN.match(timeframe)
    if not match:
        # Not every timeframe string Twelve Data accepts needs to be
        # parseable here (e.g. month-based intervals) -- it only matters
        # for the topup path below, which simply skips topping up if it
        # can't compute a step, rather than raising.
        raise ValueError(f"Cannot compute a timedelta for timeframe {timeframe!r}")
    quantity, unit = match.groups()
    return timedelta(**{_UNIT_TO_TIMEDELTA_KWARGS[unit]: int(quantity)})


class TwelveDataProvider(MarketDataProvider):
    """Market data provider backed by the Twelve Data REST API."""

    BASE_URL = "https://api.twelvedata.com"
    ENDPOINT = "/time_series"
    TIMEOUT = 10.0
    MAX_TOPUP_ATTEMPTS = 2
    TOPUP_SLEEP_SECONDS = 1.0

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
        end_date: str | None = None,
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
        end_date:
            Optional ``"YYYY-MM-DD HH:MM:SS"`` string. When provided,
            fetches ``limit`` candles ending at this point in the past
            instead of the most recent ``limit`` candles. Used for
            paginating further back in history than a single request
            can return.

        Returns
        -------
        list[MarketCandle]
            Validated candles sorted from oldest to newest. A record whose
            OHLC values fail MarketCandle's logical-relationship checks
            (e.g. high below open/close) is skipped and logged rather than
            raised -- this is tolerated because it has been observed in
            practice on the newest, still-forming candle in a live feed
            (its high/low can momentarily lag behind a tick that has
            already moved open/close), not because the validation itself
            is being loosened. A record with a bad field *type*, a bad
            datetime, or a response that is malformed at the API level
            still raises, since those indicate a broken response shape
            rather than one noisy tick.
        RuntimeError
            If Twelve Data returns an API-level error, or a record has a
            malformed field type or datetime.
        httpx.HTTPError
            If the HTTP request itself fails.
        """
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be a positive integer")

        if limit <= 0:
            raise ValueError("limit must be a positive integer")

        candles, raw_record_count = self._fetch_page(
            symbol=symbol, timeframe=timeframe, limit=limit, end_date=end_date,
        )

        # MarketDataService's pagination (get_historical_candles) treats a
        # short return from this method as "the provider has no more
        # history," and stops. That's only true when the API itself
        # returned fewer raw records than asked (raw_record_count <
        # limit) -- not when the API returned a full page but validation
        # dropped one or more candles from it. Left uncorrected, a single
        # skipped candle in an early page would make a 20,000-candle
        # fetch silently stop at ~5,000, which is worse than the crash
        # this replaced. So: only when the API actually had enough raw
        # records, make a small, bounded number of follow-up requests
        # further back in time to make up the shortfall, keeping this
        # method's short-return-means-end-of-history contract intact for
        # callers. If the API itself ran out of data (raw_record_count <
        # limit), that is returned exactly as before -- no topup.
        attempts = 0
        while (
            len(candles) < limit
            and raw_record_count == limit
            and attempts < self.MAX_TOPUP_ATTEMPTS
        ):
            if not candles:
                # No surviving candle to anchor the next window on (every
                # record in the page so far failed validation) -- rather
                # than guess at a timestamp, stop here and return what we
                # have (possibly empty), same as if topup didn't exist.
                break

            attempts += 1
            shortfall = limit - len(candles)

            try:
                step = _timeframe_to_timedelta(timeframe)
            except ValueError:
                # Can't compute how far back to page for this timeframe
                # (e.g. a month-based interval) -- stop topping up rather
                # than guess; the caller gets a slightly short result,
                # same as if this topup mechanism didn't exist.
                break

            earliest = min(candle.timestamp for candle in candles)
            topup_end_date = (earliest - step).strftime("%Y-%m-%d %H:%M:%S")

            logger.warning(
                "Topping up %d candle(s) for %s %s after validation skip "
                "(attempt %d/%d)...",
                shortfall, symbol, timeframe, attempts, self.MAX_TOPUP_ATTEMPTS,
            )
            time.sleep(self.TOPUP_SLEEP_SECONDS)

            more_candles, raw_record_count = self._fetch_page(
                symbol=symbol, timeframe=timeframe, limit=shortfall, end_date=topup_end_date,
            )

            existing_timestamps = {candle.timestamp for candle in candles}
            candles.extend(
                candle for candle in more_candles if candle.timestamp not in existing_timestamps
            )

        candles.sort(key=lambda candle: candle.timestamp)

        return candles

    def _fetch_page(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        end_date: str | None,
    ) -> tuple[list[MarketCandle], int]:
        """Request one page from Twelve Data and parse it into candles.

        Returns both the successfully-parsed candles and the raw record
        count the API returned (before any validation skips), so the
        caller can tell a validation skip apart from the API itself
        having fewer records -- see get_candles for why that distinction
        matters.
        """
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "outputsize": limit,
            "timezone": "UTC",
        }

        if end_date is not None:
            params["end_date"] = end_date

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

        candles: list[MarketCandle] = []
        skipped_count = 0

        for record in values:
            try:
                candles.append(
                    self._record_to_candle(
                        symbol=symbol,
                        timeframe=timeframe,
                        record=record,
                    ),
                )
            except ValidationError as exc:
                # Only the OHLC logical-relationship / positive-price
                # checks land here (MarketCandle's own validators) --
                # _record_to_candle already converts bad field types or a
                # bad datetime into RuntimeError below, before
                # MarketCandle is ever constructed, and those still raise.
                skipped_count += 1
                logger.warning(
                    "Skipping malformed candle for %s %s at %s: %s",
                    symbol, timeframe, record.get("datetime", "<unknown>"), exc,
                )

        if skipped_count:
            logger.warning(
                "Skipped %d of %d candle(s) for %s %s due to invalid OHLC data",
                skipped_count, len(values), symbol, timeframe,
            )

        candles.sort(key=lambda candle: candle.timestamp)

        return candles, len(values)

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