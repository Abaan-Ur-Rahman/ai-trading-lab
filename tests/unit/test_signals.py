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


def test_ema_crossover_not_yet_implemented(
    sample_dataframe: pd.DataFrame,
) -> None:
    """ema_crossover has no logic yet."""
    generator = SignalGenerator()

    with pytest.raises(NotImplementedError):
        generator.ema_crossover(sample_dataframe)


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