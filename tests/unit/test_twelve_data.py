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

def test_request_includes_end_date_when_provided(
    provider: TwelveDataProvider,
    sample_response: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """end_date should be included in the request params when provided."""
    seen: dict[str, Any] = {}

    def fake_get(url: str, params: dict[str, Any], headers: dict[str, str], timeout: float) -> Mock:
        seen["params"] = params
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = sample_response
        return response

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", fake_get)

    provider.get_candles("BTC/USD", "1h", limit=3, end_date="2023-06-01 00:00:00")

    assert seen["params"]["end_date"] == "2023-06-01 00:00:00"


def test_request_omits_end_date_when_not_provided(
    provider: TwelveDataProvider,
    sample_response: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """end_date should not appear in request params when not given (unchanged default behavior)."""
    seen: dict[str, Any] = {}

    def fake_get(url: str, params: dict[str, Any], headers: dict[str, str], timeout: float) -> Mock:
        seen["params"] = params
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = sample_response
        return response

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", fake_get)

    provider.get_candles("BTC/USD", "1h", limit=3)

    assert "end_date" not in seen["params"]

def test_malformed_ohlc_record_is_skipped_not_raised(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record failing MarketCandle's OHLC-relationship checks is skipped,
    not raised -- observed in practice on a still-forming live candle."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "ok",
        "values": [
            {
                "datetime": "2024-01-01 10:00:00",
                "open": "100.0", "high": "105.0", "low": "95.0", "close": "103.0", "volume": "1250.5",
            },
            {
                # high (1.091) below open/close -- physically impossible,
                # the exact shape of the real failure this guards against.
                "datetime": "2024-01-01 11:00:00",
                "open": "1.095", "high": "1.091", "low": "1.090", "close": "1.094", "volume": None,
            },
        ],
    }

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", lambda *args, **kwargs: response)

    candles = provider.get_candles("EUR/USD", "1h", limit=2)

    assert len(candles) == 1
    assert candles[0].timestamp == datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)


def test_topup_requests_more_when_api_had_a_full_page_but_one_record_was_skipped(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the API returns exactly `limit` raw records but one fails
    validation, get_candles should request more (not return short) so a
    validation skip never looks like 'the API ran out of history' to
    MarketDataService's pagination."""
    first_response = Mock()
    first_response.raise_for_status.return_value = None
    first_response.json.return_value = {
        "status": "ok",
        "values": [
            {
                "datetime": "2024-01-01 11:00:00",
                "open": "100.0", "high": "105.0", "low": "95.0", "close": "103.0", "volume": "1250.5",
            },
            {
                # malformed -- high below open/close
                "datetime": "2024-01-01 12:00:00",
                "open": "101.0", "high": "99.0", "low": "95.0", "close": "100.5", "volume": "1300.0",
            },
        ],
    }
    topup_response = Mock()
    topup_response.raise_for_status.return_value = None
    topup_response.json.return_value = {
        "status": "ok",
        "values": [
            {
                "datetime": "2024-01-01 10:00:00",
                "open": "99.0", "high": "104.0", "low": "94.0", "close": "102.0", "volume": "1200.0",
            },
        ],
    }

    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any], headers: dict[str, str], timeout: float) -> Mock:
        calls.append(params)
        return first_response if len(calls) == 1 else topup_response

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", fake_get)
    monkeypatch.setattr("data.providers.twelve_data.time.sleep", lambda seconds: None)

    candles = provider.get_candles("EUR/USD", "1h", limit=2)

    assert len(candles) == 2
    assert [c.timestamp for c in candles] == sorted(c.timestamp for c in candles)
    assert len(calls) == 2
    # The topup request should ask for exactly the shortfall (1), ending
    # before the earliest candle already in hand.
    assert calls[1]["outputsize"] == 1
    assert calls[1]["end_date"] == "2024-01-01 10:00:00"


def test_no_topup_when_api_itself_returned_fewer_than_requested(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short raw response (no skips involved) means the API itself ran
    out of data -- get_candles must NOT top this up, preserving
    MarketDataService's 'short return means end of history' contract."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "ok",
        "values": [
            {
                "datetime": "2024-01-01 10:00:00",
                "open": "100.0", "high": "105.0", "low": "95.0", "close": "103.0", "volume": "1250.5",
            },
        ],
    }

    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any], headers: dict[str, str], timeout: float) -> Mock:
        calls.append(params)
        return response

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", fake_get)

    candles = provider.get_candles("EUR/USD", "1h", limit=5)

    assert len(candles) == 1
    assert len(calls) == 1  # no topup attempted


def test_topup_does_not_guess_when_nothing_survived_to_anchor_on(
    provider: TwelveDataProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the only record(s) in the page all fail validation, there is no
    surviving timestamp to page backward from -- get_candles must stop
    rather than guess, returning a short (possibly empty) result instead
    of looping or crashing on an empty min()."""
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "status": "ok",
        "values": [
            {
                "datetime": "2024-01-01 12:00:00",
                "open": "101.0", "high": "99.0", "low": "95.0", "close": "100.5", "volume": "1300.0",
            },
        ],
    }

    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any], headers: dict[str, str], timeout: float) -> Mock:
        calls.append(params)
        return response

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", fake_get)
    monkeypatch.setattr("data.providers.twelve_data.time.sleep", lambda seconds: None)

    candles = provider.get_candles("EUR/USD", "1h", limit=1)

    assert candles == []
    assert len(calls) == 1  # no topup attempted -- nothing to anchor it on

    def multi_record_response() -> Mock:
        resp = Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "status": "ok",
            "values": [
                {
                    "datetime": "2024-01-01 11:00:00",
                    "open": "100.0", "high": "105.0", "low": "95.0", "close": "103.0", "volume": "1250.5",
                },
                {
                    "datetime": "2024-01-01 12:00:00",
                    "open": "101.0", "high": "99.0", "low": "95.0", "close": "100.5", "volume": "1300.0",
                },
            ],
        }
        return resp

    calls_2: list[dict[str, Any]] = []

    def fake_get_2(url: str, params: dict[str, Any], headers: dict[str, str], timeout: float) -> Mock:
        calls_2.append(params)
        return multi_record_response()

    monkeypatch.setattr("data.providers.twelve_data.httpx.get", fake_get_2)

    # Here one record DOES survive, so topup should keep retrying (using
    # that surviving candle's timestamp as the anchor) up to the bound,
    # rather than stopping after one attempt.
    candles = provider.get_candles("EUR/USD", "1h", limit=2)

    assert len(candles) == 1
    assert len(calls_2) == 1 + TwelveDataProvider.MAX_TOPUP_ATTEMPTS