"""Unit tests for TwelveDataProvider."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from data.models import MarketCandle
from data.providers.twelve_data import TwelveDataProvider


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> TwelveDataProvider:
    """Create a provider instance with a mocked API key."""
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test-api-key")
    return TwelveDataProvider()


@pytest.fixture
def sample_response() -> dict[str, Any]:
    """Return a realistic mock response payload from Twelve Data."""
    return {
        "status": "ok",
        "values": [
            {
                "datetime": "2024-01-01 10:00:00",
                "open": "100.0",
                "high": "105.0",
                "low": "95.0",
                "close": "103.0",
                "volume": "1250.5",
            },
            {
                "datetime": "2024-01-01 11:00:00",
                "open": "101.0",
                "high": "106.0",
                "low": "96.0",
                "close": "104.0",
                "volume": "1300.0",
            },
            {
                "datetime": "2024-01-01 12:00:00",
                "open": "102.0",
                "high": "107.0",
                "low": "97.0",
                "close": "105.0",
                "volume": "1350.0",
            },
        ],
    }


def test_provider_initializes_when_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider should initialize when the API key is available."""
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test-api-key")

    provider = TwelveDataProvider()

    assert provider._api_key == "test-api-key"


def test_provider_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider should raise when API key env var is missing."""
    monkeypatch.setattr("data.providers.twelve_data.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TWELVE_DATA_API_KEY"):
        TwelveDataProvider()


@pytest.mark.parametrize("limit", [0, -1])
def test_get_candles_rejects_non_positive_limits(limit: int, provider: TwelveDataProvider) -> None:
    """Limits must be positive integers."""
    with pytest.raises(ValueError, match="positive integer"):
        provider.get_candles("BTC/USD", "1h", limit=limit)


def test_get_candles_rejects_non_integer_limits(provider: TwelveDataProvider) -> None:
    """Non-integer limits should be rejected."""
    with pytest.raises(ValueError, match="positive integer"):
        provider.get_candles("BTC/USD", "1h", limit="10")


def test_valid_response_is_converted_to_market_candles(
    provider: TwelveDataProvider,
    sample_response: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid response should be converted into MarketCandle objects."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = sample_response

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", lambda *args, **kwargs: response)

    candles = provider.get_candles("BTC/USD", "1h", limit=3)

    assert len(candles) == 3
    assert all(isinstance(candle, MarketCandle) for candle in candles)
    assert candles[0].symbol == "BTC/USD"
    assert candles[0].timeframe == "1h"
    assert candles[0].timestamp == datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert candles[0].open == 100.0
    assert candles[0].high == 105.0
    assert candles[0].low == 95.0
    assert candles[0].close == 103.0
    assert candles[0].volume == 1250.5


def test_returned_candles_are_sorted_oldest_first(
    provider: TwelveDataProvider,
    sample_response: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candles should be sorted by timestamp ascending."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = sample_response

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", lambda *args, **kwargs: response)

    candles = provider.get_candles("BTC/USD", "1h", limit=3)

    timestamps = [candle.timestamp for candle in candles]
    assert timestamps == sorted(timestamps)


def test_request_includes_symbol_interval_outputsize_and_timezone(
    provider: TwelveDataProvider,
    sample_response: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request parameters should include the expected payload."""
    seen: dict[str, Any] = {}

    def fake_get(
        url: str,
        params: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> Mock:
        seen["url"] = url
        seen["params"] = params
        seen["headers"] = headers
        seen["timeout"] = timeout
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = sample_response
        return response

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", fake_get)

    provider.get_candles("BTC/USD", "1h", limit=3)

    assert seen["url"] == "https://api.twelvedata.com/time_series"
    assert seen["params"]["symbol"] == "BTC/USD"
    assert seen["params"]["interval"] == "1h"
    assert seen["params"]["outputsize"] == 3
    assert seen["params"]["timezone"] == "UTC"
    assert seen["headers"]["Authorization"] == "apikey test-api-key"
    assert seen["timeout"] == provider.TIMEOUT


def test_api_error_response_raises_runtime_error(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API errors should raise RuntimeError with a clear message."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"status": "error", "message": "bad request"}

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="Twelve Data API error"):
        provider.get_candles("BTC/USD", "1h", limit=3)


def test_missing_values_key_raises_runtime_error(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing values should raise RuntimeError."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"status": "ok"}

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="'values' must be a list"):
        provider.get_candles("BTC/USD", "1h", limit=3)


def test_invalid_ohlc_data_raises_error(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed OHLC values should raise an error."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "ok",
        "values": [{"datetime": "2024-01-01 10:00:00", "open": "bad", "high": "105", "low": "95", "close": "103"}],
    }

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="Invalid OHLC data received from Twelve Data"):
        provider.get_candles("BTC/USD", "1h", limit=3)


def test_invalid_datetime_data_raises_error(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid datetime strings should raise an error."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "ok",
        "values": [{"datetime": "not-a-date", "open": "100", "high": "105", "low": "95", "close": "103"}],
    }

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="Invalid datetime format received from Twelve Data"):
        provider.get_candles("BTC/USD", "1h", limit=3)


def test_http_error_response_is_handled_correctly(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP errors should be surfaced as RuntimeError."""
    response = Mock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "request failed",
        request=Mock(),
        response=Mock(status_code=500),
    )

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", lambda *args, **kwargs: response)

    with pytest.raises(httpx.HTTPStatusError):
        provider.get_candles("BTC/USD", "1h", limit=3)
