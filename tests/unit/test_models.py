"""Unit tests for LogisticRegressionModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.dataset import CLASS_NAMES, FEATURE_COLUMNS
from ml.models.logistic_regression import LogisticRegressionModel


@pytest.fixture
def small_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small synthetic dataset covering all three classes."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    y = pd.Series([-1] * 20 + [0] * 20 + [1] * 20, name="label")
    return X, y


def test_predict_raises_before_fit(small_training_data) -> None:
    """predict() before fit() should raise RuntimeError."""
    X_train, _ = small_training_data
    model = LogisticRegressionModel()

    with pytest.raises(RuntimeError):
        model.predict(X_train)


def test_predict_proba_raises_before_fit(small_training_data) -> None:
    """predict_proba() before fit() should raise RuntimeError."""
    X_train, _ = small_training_data
    model = LogisticRegressionModel()

    with pytest.raises(RuntimeError):
        model.predict_proba(X_train)


def test_predict_returns_only_valid_classes(small_training_data) -> None:
    """predict() should only return values in {-1, 0, 1}."""
    X_train, y_train = small_training_data
    model = LogisticRegressionModel()
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    assert set(np.unique(predictions)).issubset({-1, 0, 1})


def test_predict_proba_rows_sum_to_one(small_training_data) -> None:
    """Each row of predict_proba() should sum to 1."""
    X_train, y_train = small_training_data
    model = LogisticRegressionModel()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_train)

    assert proba.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(X_train)))


def test_predict_proba_column_order(small_training_data) -> None:
    """predict_proba() columns must always be [SELL, HOLD, BUY]."""
    X_train, y_train = small_training_data
    model = LogisticRegressionModel()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_train)

    assert list(proba.columns) == CLASS_NAMES


def test_same_random_state_is_reproducible(small_training_data) -> None:
    """Two models with the same random_state should predict identically."""
    X_train, y_train = small_training_data

    model_a = LogisticRegressionModel(random_state=42)
    model_a.fit(X_train, y_train)

    model_b = LogisticRegressionModel(random_state=42)
    model_b.fit(X_train, y_train)

    np.testing.assert_array_equal(model_a.predict(X_train), model_b.predict(X_train))


def test_class_weight_defaults_to_none() -> None:
    """class_weight should default to None, not 'balanced'."""
    model = LogisticRegressionModel()

    assert model._class_weight is None


def test_class_weight_balanced_is_supported(small_training_data) -> None:
    """class_weight='balanced' must remain a fully supported configuration."""
    X_train, y_train = small_training_data
    model = LogisticRegressionModel(class_weight="balanced")
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    assert set(np.unique(predictions)).issubset({-1, 0, 1})


def test_predict_proba_reindexes_regardless_of_classes_order(
    small_training_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    """predict_proba() must reorder columns even if sklearn's classes_ order differs.

    This simulates a hypothetical case where the underlying sklearn model
    discovers classes in a non-ascending order, to prove the wrapper's
    reindexing logic -- not sklearn's incidental sorting behavior -- is
    what guarantees the fixed [SELL, HOLD, BUY] column order.
    """
    X_train, y_train = small_training_data
    model = LogisticRegressionModel()
    model.fit(X_train, y_train)

    # Simulate classes_ coming back in a different order: [BUY, SELL, HOLD]
    monkeypatch.setattr(model._model, "classes_", np.array([1, -1, 0]))
    fake_raw_proba = np.array([[0.2, 0.5, 0.3]])  # BUY=0.2, SELL=0.5, HOLD=0.3
    monkeypatch.setattr(model._model, "predict_proba", lambda X: fake_raw_proba)

    result = model.predict_proba(X_train.iloc[[0]])

    assert list(result.columns) == CLASS_NAMES
    assert result.iloc[0]["SELL"] == pytest.approx(0.5)
    assert result.iloc[0]["HOLD"] == pytest.approx(0.3)
    assert result.iloc[0]["BUY"] == pytest.approx(0.2)