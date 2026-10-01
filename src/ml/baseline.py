"""Baseline predictors for comparison against trained ML models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.dataset import CLASS_LABELS, CLASS_NAMES
from strategy.signals import Signal, SignalGenerator


class MajorityClassBaseline:
    """Predicts the most frequent class seen during training, always.

    This is the statistical floor: any real model that can't beat this
    isn't learning anything useful.
    """

    def __init__(self) -> None:
        self._majority_class: int | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Store the most frequent class in y_train."""
        self._majority_class = int(y_train.value_counts().idxmax())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the majority class, repeated for every row in X."""
        if self._majority_class is None:
            raise RuntimeError("MajorityClassBaseline must be fit before predict")

        return np.full(len(X), self._majority_class)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a one-hot probability distribution on the majority class."""
        if self._majority_class is None:
            raise RuntimeError("MajorityClassBaseline must be fit before predict_proba")

        probabilities = pd.DataFrame(0.0, index=X.index, columns=CLASS_NAMES)
        majority_name = CLASS_NAMES[CLASS_LABELS.index(self._majority_class)]
        probabilities[majority_name] = 1.0

        return probabilities


class RuleBasedBaseline:
    """Wraps the existing SignalGenerator.combined_signal as a baseline.

    Unlike MajorityClassBaseline, this needs raw OHLCV data (not engineered
    features), since it calls the existing, already-tested strategy layer
    directly rather than reimplementing any signal logic. There is no fit()
    -- the rules are fixed, not learned from training data.

    For each row, combined_signal is recomputed using an expanding window
    of all OHLCV history up to and including that row. Rows without enough
    history for the underlying indicators (EMA/RSI/MACD warm-up) default to
    HOLD with confidence 0.0 -- treated as "no signal yet", not an error.

    This recomputes indicators from scratch at every row rather than
    vectorizing, so it is the slowest component built so far. Acceptable
    for research/backtest-scale data; would need revisiting for live use.
    """

    def __init__(self, signal_generator: SignalGenerator | None = None) -> None:
        self._signal_generator = signal_generator or SignalGenerator()

    def predict_proba(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return a one-hot probability distribution from combined_signal per row."""
        rows = []

        for position in range(len(ohlcv)):
            window = ohlcv.iloc[: position + 1]

            try:
                result = self._signal_generator.combined_signal(window)
                direction = result.direction
            except ValueError:
                direction = Signal.HOLD

            row = {name: 0.0 for name in CLASS_NAMES}
            row[direction.value] = 1.0
            rows.append(row)

        return pd.DataFrame(rows, index=ohlcv.index, columns=CLASS_NAMES)

    def predict(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """Return a combined_signal-derived label for every row in ohlcv."""
        probabilities = self.predict_proba(ohlcv)
        winning_names = probabilities.idxmax(axis=1)

        return np.array([CLASS_LABELS[CLASS_NAMES.index(name)] for name in winning_names])