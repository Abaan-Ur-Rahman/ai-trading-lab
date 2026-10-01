"""Unit tests for baseline predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baseline import MajorityClassBaseline, RuleBasedBaseline
from strategy.signals import SignalGenerator


def test_majority_class_baseline_predicts_most_frequent_class() -> None:
    """The baseline should always predict whichever class was most frequent in training."""
    baseline = MajorityClassBaseline()
    y_train = pd.Series([1, 1, 1, 0, 0, -1])

    baseline.fit(pd.DataFrame(index=y_train.index), y_train)

    X = pd.DataFrame(index=range(3))
    predictions = baseline.predict(X)

    assert (predictions == 1).all()


def test_majority_class_baseline_predict_proba_is_one_hot() -> None:
    """predict_proba should put 1.0 on the majority class and 0.0 elsewhere."""
    baseline = MajorityClassBaseline()
    y_train = pd.Series([0, 0, 0, 1, -1])

    baseline.fit(pd.DataFrame(index=y_train.index), y_train)

    X = pd.DataFrame(index=range(2))
    probabilities = baseline.predict_proba(X)

    assert (probabilities["HOLD"] == 1.0).all()
    assert (probabilities["SELL"] == 0.0).all()
    assert (probabilities["BUY"] == 0.0).all()
    assert list(probabilities.columns) == ["SELL", "HOLD", "BUY"]


def test_majority_class_baseline_requires_fit_before_predict() -> None:
    """Calling predict before fit should raise, not silently return garbage."""
    baseline = MajorityClassBaseline()

    with pytest.raises(RuntimeError):
        baseline.predict(pd.DataFrame(index=range(3)))


def test_majority_class_baseline_requires_fit_before_predict_proba() -> None:
    """Calling predict_proba before fit should raise, not silently return garbage."""
    baseline = MajorityClassBaseline()

    with pytest.raises(RuntimeError):
        baseline.predict_proba(pd.DataFrame(index=range(3)))


def _make_ohlcv(rows: int) -> pd.DataFrame:
    """Flat then trending close series: deterministic, no RNG dependency."""
    flat = [100.0] * 20
    trend = [100 + i * 1.5 for i in range(1, 21)]
    close = pd.Series((flat + trend)[:rows], dtype=float)

    return pd.DataFrame({"close": close, "high": close + 1, "low": close - 1})


def test_rule_based_baseline_matches_combined_signal_directly() -> None:
    """The last row's result must exactly match calling combined_signal on the full data.

    This is the cross-check the design called for: RuleBasedBaseline must
    not reimplement any signal logic, only call the existing, already-tested
    SignalGenerator.
    """
    ohlcv = _make_ohlcv(rows=40)
    generator = SignalGenerator()

    expected = generator.combined_signal(ohlcv)

    baseline = RuleBasedBaseline(signal_generator=generator)
    probabilities = baseline.predict_proba(ohlcv)
    last_row = probabilities.iloc[-1]

    assert last_row[expected.direction.value] == 1.0
    assert last_row.sum() == 1.0


def test_rule_based_baseline_defaults_to_hold_when_not_enough_history() -> None:
    """Rows without enough history for the underlying indicators should default to HOLD."""
    ohlcv = _make_ohlcv(rows=10)  # well below the 35-row minimum for default MACD params

    baseline = RuleBasedBaseline()
    probabilities = baseline.predict_proba(ohlcv)

    assert (probabilities["HOLD"] == 1.0).all()


def test_rule_based_baseline_predict_matches_predict_proba_argmax() -> None:
    """predict() should return the class predict_proba assigned probability 1.0 to."""
    ohlcv = _make_ohlcv(rows=40)

    baseline = RuleBasedBaseline()
    probabilities = baseline.predict_proba(ohlcv)
    predictions = baseline.predict(ohlcv)

    for position, label in enumerate(predictions):
        name_for_label = {-1: "SELL", 0: "HOLD", 1: "BUY"}[label]
        assert probabilities.iloc[position][name_for_label] == 1.0