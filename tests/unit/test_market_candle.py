"""Unit tests for MarketCandle."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.models import MarketCandle


@pytest.fixture
def valid_candle_data() -> dict[str, object]:
    """Return a valid MarketCandle payload for reuse in tests."""
    return {
        "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        "symbol": " btc-usd ",
        "timeframe": "1h",
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 103.0,
        "volume": 1250.5,
    }


def test_valid_candle_is_accepted(valid_candle_data: dict[str, object]) -> None:
    """A well-formed candle should parse successfully."""
    candle = MarketCandle(**valid_candle_data)

    assert candle.symbol == "BTC-USD"
    assert candle.timeframe == "1h"
    assert candle.open == 100.0
    assert candle.high == 105.0
    assert candle.low == 95.0
    assert candle.close == 103.0
    assert candle.volume == 1250.5


def test_symbol_is_normalized_and_whitespace_is_stripped(
    valid_candle_data: dict[str, object],
) -> None:
    """Symbol should be converted to uppercase and trimmed."""
    candle = MarketCandle(**valid_candle_data)

    assert candle.symbol == "BTC-USD"


def test_volume_can_be_none(valid_candle_data: dict[str, object]) -> None:
    """Volume should be accepted as None."""
    payload = dict(valid_candle_data)
    payload["volume"] = None

    candle = MarketCandle(**payload)

    assert candle.volume is None


@pytest.mark.parametrize("price_field", ["open", "high", "low", "close"])
def test_negative_price_is_rejected(price_field: str, valid_candle_data: dict[str, object]) -> None:
    """Negative prices should be rejected."""
    payload = dict(valid_candle_data)
    payload[price_field] = -1.0

    with pytest.raises(ValueError):
        MarketCandle(**payload)


def test_zero_price_is_rejected(valid_candle_data: dict[str, object]) -> None:
    """Zero prices should be rejected."""
    payload = dict(valid_candle_data)
    payload["open"] = 0.0

    with pytest.raises(ValueError):
        MarketCandle(**payload)


def test_high_less_than_low_is_rejected(valid_candle_data: dict[str, object]) -> None:
    """High must be greater than or equal to low."""
    payload = dict(valid_candle_data)
    payload["high"] = 90.0

    with pytest.raises(ValueError):
        MarketCandle(**payload)


def test_high_less_than_open_is_rejected(valid_candle_data: dict[str, object]) -> None:
    """High must be at least as large as open."""
    payload = dict(valid_candle_data)
    payload["high"] = 99.0

    with pytest.raises(ValueError):
        MarketCandle(**payload)


def test_high_less_than_close_is_rejected(valid_candle_data: dict[str, object]) -> None:
    """High must be at least as large as close."""
    payload = dict(valid_candle_data)
    payload["high"] = 102.0

    with pytest.raises(ValueError):
        MarketCandle(**payload)


def test_low_greater_than_open_is_rejected(valid_candle_data: dict[str, object]) -> None:
    """Low must be no larger than open."""
    payload = dict(valid_candle_data)
    payload["low"] = 101.0

    with pytest.raises(ValueError):
        MarketCandle(**payload)


def test_low_greater_than_close_is_rejected(valid_candle_data: dict[str, object]) -> None:
    """Low must be no larger than close."""
    payload = dict(valid_candle_data)
    payload["low"] = 104.0

    with pytest.raises(ValueError):
        MarketCandle(**payload)


def test_empty_symbol_is_rejected(valid_candle_data: dict[str, object]) -> None:
    """An empty symbol should be rejected."""
    payload = dict(valid_candle_data)
    payload["symbol"] = ""

    with pytest.raises(ValueError):
        MarketCandle(**payload)


def test_empty_timeframe_is_rejected(valid_candle_data: dict[str, object]) -> None:
    """An empty timeframe should be rejected."""
    payload = dict(valid_candle_data)
    payload["timeframe"] = ""

    with pytest.raises(ValueError):
        MarketCandle(**payload)
