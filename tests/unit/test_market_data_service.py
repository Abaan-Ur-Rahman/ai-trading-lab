from unittest.mock import Mock

import pytest

from data.models import MarketCandle
from data.providers.base import MarketDataProvider
from data.services.market_data import MarketDataService


def test_service_delegates_to_provider():
    provider = Mock(spec=MarketDataProvider)

    candles = [
        MarketCandle(
            timestamp="2024-01-01T00:00:00Z",
            symbol="BTC/USD",
            timeframe="1h",
            open=100,
            high=110,
            low=90,
            close=105,
            volume=1000,
        )
    ]

    provider.get_candles.return_value = candles

    service = MarketDataService(provider)

    result = service.get_recent_candles(
        "BTC/USD",
        "1h",
        limit=1,
    )

    assert result == candles

    provider.get_candles.assert_called_once_with(
        symbol="BTC/USD",
        timeframe="1h",
        limit=1,
    )


def _candle(ts: str) -> MarketCandle:
    return MarketCandle(
        timestamp=ts, symbol="XAU/USD", timeframe="1h",
        open=100, high=101, low=99, close=100.5, volume=None,
    )


def test_get_historical_candles_paginates_until_total_reached():
    provider = Mock(spec=MarketDataProvider)

    page1 = [_candle("2024-01-10T20:00:00Z"), _candle("2024-01-10T21:00:00Z")]
    page2 = [_candle("2024-01-10T18:00:00Z"), _candle("2024-01-10T19:00:00Z")]
    provider.get_candles.side_effect = [page1, page2]

    service = MarketDataService(provider)
    sleeps = []

    result = service.get_historical_candles(
        "XAU/USD", "1h", total_candles=4, page_size=2, sleep_fn=sleeps.append,
    )

    assert len(result) == 4
    assert [c.timestamp for c in result] == sorted(c.timestamp for c in result)
    assert provider.get_candles.call_count == 2
    assert sleeps == [8.0]


def test_get_historical_candles_stops_when_provider_returns_fewer_than_requested():
    provider = Mock(spec=MarketDataProvider)

    full_page = [_candle("2024-01-10T20:00:00Z"), _candle("2024-01-10T21:00:00Z")]
    short_page = [_candle("2024-01-10T19:00:00Z")]
    provider.get_candles.side_effect = [full_page, short_page]

    service = MarketDataService(provider)

    result = service.get_historical_candles(
        "XAU/USD", "1h", total_candles=10, page_size=2, sleep_fn=lambda s: None,
    )

    assert len(result) == 3
    assert provider.get_candles.call_count == 2


def test_get_historical_candles_rejects_non_positive_total_candles():
    provider = Mock(spec=MarketDataProvider)
    service = MarketDataService(provider)

    with pytest.raises(ValueError):
        service.get_historical_candles("XAU/USD", "1h", total_candles=0)


def test_get_historical_candles_rejects_non_positive_page_size():
    provider = Mock(spec=MarketDataProvider)
    service = MarketDataService(provider)

    with pytest.raises(ValueError):
        service.get_historical_candles("XAU/USD", "1h", total_candles=10, page_size=0)


def test_get_historical_candles_rejects_unsupported_timeframe():
    provider = Mock(spec=MarketDataProvider)
    service = MarketDataService(provider)

    with pytest.raises(ValueError, match="Unsupported timeframe"):
        service.get_historical_candles("XAU/USD", "1month", total_candles=10)


def test_get_historical_candles_passes_end_date_to_second_request():
    provider = Mock(spec=MarketDataProvider)

    page1 = [_candle("2024-01-10T20:00:00Z"), _candle("2024-01-10T21:00:00Z")]
    page2 = [_candle("2024-01-10T18:00:00Z"), _candle("2024-01-10T19:00:00Z")]
    provider.get_candles.side_effect = [page1, page2]

    service = MarketDataService(provider)
    service.get_historical_candles("XAU/USD", "1h", total_candles=4, page_size=2, sleep_fn=lambda s: None)

    first_call_kwargs = provider.get_candles.call_args_list[0].kwargs
    second_call_kwargs = provider.get_candles.call_args_list[1].kwargs

    assert first_call_kwargs["end_date"] is None
    assert second_call_kwargs["end_date"] == "2024-01-10 19:00:00"