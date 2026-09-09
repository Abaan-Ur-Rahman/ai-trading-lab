"""Unit tests for the SignalGenerator skeleton."""

from __future__ import annotations

import pandas as pd
import pytest

from strategy.signals import Signal, SignalGenerator, SignalResult


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a small OHLCV dataframe."""

    return pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104],
        }
    )


def test_signal_generator_exists() -> None:
    """SignalGenerator can be instantiated."""
    generator = SignalGenerator()

    assert generator is not None


def test_signal_result_rejects_out_of_range_confidence() -> None:
    """SignalResult should reject confidence outside 0-1."""

    with pytest.raises(ValueError):
        SignalResult(direction=Signal.BUY, confidence=1.5)


def test_rsi_signal_detects_buy() -> None:
    """RSI at or below oversold should signal BUY."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [100, 98, 96, 94, 92, 90, 88, 86]},
    )

    result = generator.rsi_signal(dataframe, length=6)

    assert result.direction == Signal.BUY
    assert result.confidence == 1.0


def test_rsi_signal_detects_sell() -> None:
    """RSI at or above overbought should signal SELL."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [100, 102, 104, 106, 108, 110, 112, 114]},
    )

    result = generator.rsi_signal(dataframe, length=6)

    assert result.direction == Signal.SELL
    assert result.confidence == 1.0


def test_rsi_signal_detects_hold() -> None:
    """RSI between thresholds should signal HOLD."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [100, 101, 100, 101, 100, 101, 100, 101]},
    )

    result = generator.rsi_signal(dataframe, length=6)

    assert result.direction == Signal.HOLD
    assert result.confidence == 0.0


def test_rsi_signal_requires_oversold_less_than_overbought() -> None:
    """oversold must be strictly less than overbought."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame({"close": [100] * 10})

    with pytest.raises(ValueError):
        generator.rsi_signal(dataframe, oversold=70, overbought=30)

def test_rsi_signal_raises_on_flat_price_data() -> None:
    """Zero price movement produces an all-NaN RSI; this should raise, not silently HOLD."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame({"close": [100.0] * 10})

    with pytest.raises(ValueError):
        generator.rsi_signal(dataframe, length=6)


def test_macd_signal_detects_buy() -> None:
    """MACD line crossing above signal line should signal BUY."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [110, 108, 106, 104, 102, 100, 98, 130]},
    )

    result = generator.macd_signal(dataframe, fast=3, slow=6, signal=2)

    assert result.direction == Signal.BUY
    assert result.confidence == 1.0


def test_macd_signal_detects_sell() -> None:
    """MACD line crossing below signal line should signal SELL."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [90, 92, 94, 96, 98, 100, 102, 60]},
    )

    result = generator.macd_signal(dataframe, fast=3, slow=6, signal=2)

    assert result.direction == Signal.SELL
    assert result.confidence == 1.0


def test_macd_signal_detects_hold() -> None:
    """No crossover should signal HOLD."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [100] * 10},
    )

    result = generator.macd_signal(dataframe, fast=3, slow=6, signal=2)

    assert result.direction == Signal.HOLD
    assert result.confidence == 0.0


def test_macd_signal_requires_enough_data() -> None:
    """Not enough data for a full crossover comparison should raise ValueError."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [100, 101, 102, 103, 104, 105, 106]},
    )

    with pytest.raises(ValueError):
        generator.macd_signal(dataframe, fast=3, slow=6, signal=2)


def _mock_signal(direction: Signal, confidence: float = 1.0):
    """Build a stand-in signal method that always returns a fixed result."""
    def _mock(*_args, **_kwargs) -> SignalResult:
        return SignalResult(direction=direction, confidence=confidence)
    return _mock


def test_combined_signal_majority_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two BUY votes out of three should win with 2/3 confidence."""
    generator = SignalGenerator()

    monkeypatch.setattr(generator, "ema_crossover", _mock_signal(Signal.BUY))
    monkeypatch.setattr(generator, "rsi_signal", _mock_signal(Signal.BUY))
    monkeypatch.setattr(generator, "macd_signal", _mock_signal(Signal.SELL))

    dataframe = pd.DataFrame({"close": [100] * 10})

    result = generator.combined_signal(dataframe)

    assert result.direction == Signal.BUY
    assert result.confidence == pytest.approx(2 / 3)


def test_combined_signal_majority_sell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two SELL votes out of three should win with 2/3 confidence."""
    generator = SignalGenerator()

    monkeypatch.setattr(generator, "ema_crossover", _mock_signal(Signal.SELL))
    monkeypatch.setattr(generator, "rsi_signal", _mock_signal(Signal.SELL))
    monkeypatch.setattr(generator, "macd_signal", _mock_signal(Signal.BUY))

    dataframe = pd.DataFrame({"close": [100] * 10})

    result = generator.combined_signal(dataframe)

    assert result.direction == Signal.SELL
    assert result.confidence == pytest.approx(2 / 3)


def test_combined_signal_unanimous_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full agreement on HOLD should produce HOLD with full confidence."""
    generator = SignalGenerator()

    monkeypatch.setattr(generator, "ema_crossover", _mock_signal(Signal.HOLD))
    monkeypatch.setattr(generator, "rsi_signal", _mock_signal(Signal.HOLD))
    monkeypatch.setattr(generator, "macd_signal", _mock_signal(Signal.HOLD))

    dataframe = pd.DataFrame({"close": [100] * 10})

    result = generator.combined_signal(dataframe)

    assert result.direction == Signal.HOLD
    assert result.confidence == pytest.approx(1.0)


def test_combined_signal_no_majority_defaults_to_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three different directions (no majority) should default to HOLD with 0 confidence."""
    generator = SignalGenerator()

    monkeypatch.setattr(generator, "ema_crossover", _mock_signal(Signal.BUY))
    monkeypatch.setattr(generator, "rsi_signal", _mock_signal(Signal.SELL))
    monkeypatch.setattr(generator, "macd_signal", _mock_signal(Signal.HOLD))

    dataframe = pd.DataFrame({"close": [100] * 10})

    result = generator.combined_signal(dataframe)

    assert result.direction == Signal.HOLD
    assert result.confidence == 0.0

def test_ema_crossover_detects_buy() -> None:
    """A fast EMA crossing above the slow EMA should signal BUY."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [110, 108, 106, 104, 102, 100, 98, 130]},
    )

    result = generator.ema_crossover(dataframe, fast=3, slow=6)

    assert result.direction == Signal.BUY
    assert result.confidence == 1.0


def test_ema_crossover_detects_sell() -> None:
    """A fast EMA crossing below the slow EMA should signal SELL."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [90, 92, 94, 96, 98, 100, 102, 60]},
    )

    result = generator.ema_crossover(dataframe, fast=3, slow=6)

    assert result.direction == Signal.SELL
    assert result.confidence == 1.0


def test_ema_crossover_detects_hold() -> None:
    """No crossover should signal HOLD."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame(
        {"close": [100] * 10},
    )

    result = generator.ema_crossover(dataframe, fast=3, slow=6)

    assert result.direction == Signal.HOLD
    assert result.confidence == 0.0


def test_ema_crossover_requires_fast_less_than_slow() -> None:
    """fast must be strictly less than slow."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame({"close": [100] * 10})

    with pytest.raises(ValueError):
        generator.ema_crossover(dataframe, fast=10, slow=5)


def test_ema_crossover_requires_enough_rows() -> None:
    """There must be enough rows for two valid EMA points."""
    generator = SignalGenerator()

    dataframe = pd.DataFrame({"close": [100, 101, 102]})

    with pytest.raises(ValueError):
        generator.ema_crossover(dataframe, fast=3, slow=6)