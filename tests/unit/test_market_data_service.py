from unittest.mock import Mock

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