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


def test_rsi_signal_not_yet_implemented(
    sample_dataframe: pd.DataFrame,
) -> None:
    """rsi_signal has no logic yet."""
    generator = SignalGenerator()

    with pytest.raises(NotImplementedError):
        generator.rsi_signal(sample_dataframe)


def test_macd_signal_not_yet_implemented(
    sample_dataframe: pd.DataFrame,
) -> None:
    """macd_signal has no logic yet."""
    generator = SignalGenerator()

    with pytest.raises(NotImplementedError):
        generator.macd_signal(sample_dataframe)


def test_combined_signal_not_yet_implemented(
    sample_dataframe: pd.DataFrame,
) -> None:
    """combined_signal has no logic yet."""
    generator = SignalGenerator()

    with pytest.raises(NotImplementedError):
        generator.combined_signal(sample_dataframe)

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