"""Unit tests for RandomForestModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.dataset import CLASS_NAMES, FEATURE_COLUMNS
from ml.models.random_forest import RandomForestModel


@pytest.fixture
def small_training_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 90
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    y = pd.Series([-1] * 30 + [0] * 30 + [1] * 30, name="label")
    return X, y


def test_predict_raises_before_fit(small_training_data) -> None:
    X_train, _ = small_training_data
    model = RandomForestModel()

    with pytest.raises(RuntimeError):
        model.predict(X_train)


def test_predict_proba_raises_before_fit(small_training_data) -> None:
    X_train, _ = small_training_data
    model = RandomForestModel()

    with pytest.raises(RuntimeError):
        model.predict_proba(X_train)


def test_get_feature_importances_raises_before_fit() -> None:
    model = RandomForestModel()

    with pytest.raises(RuntimeError):
        model.get_feature_importances()


def test_predict_returns_only_valid_classes(small_training_data) -> None:
    X_train, y_train = small_training_data
    model = RandomForestModel()
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    assert set(np.unique(predictions)).issubset({-1, 0, 1})


def test_predict_proba_rows_sum_to_one(small_training_data) -> None:
    X_train, y_train = small_training_data
    model = RandomForestModel()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_train)

    assert proba.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(X_train)))


def test_predict_proba_column_order(small_training_data) -> None:
    X_train, y_train = small_training_data
    model = RandomForestModel()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_train)

    assert list(proba.columns) == CLASS_NAMES


def test_same_random_state_is_reproducible(small_training_data) -> None:
    X_train, y_train = small_training_data

    model_a = RandomForestModel(random_state=42)
    model_a.fit(X_train, y_train)

    model_b = RandomForestModel(random_state=42)
    model_b.fit(X_train, y_train)

    np.testing.assert_array_equal(model_a.predict(X_train), model_b.predict(X_train))


def test_defaults_are_conservative_not_sklearn_defaults() -> None:
    """max_depth/min_samples_leaf must default to the regularized values, not sklearn's."""
    model = RandomForestModel()

    assert model.get_hyperparameters() == {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_leaf": 20,
        "class_weight": None,
        "random_state": 42,
    }


def test_class_weight_balanced_is_supported(small_training_data) -> None:
    X_train, y_train = small_training_data
    model = RandomForestModel(class_weight="balanced")
    model.fit(X_train, y_train)

    predictions = model.predict(X_train)

    assert set(np.unique(predictions)).issubset({-1, 0, 1})


def test_predict_proba_reindexes_regardless_of_classes_order(
    small_training_data, monkeypatch: pytest.MonkeyPatch
) -> None:
    X_train, y_train = small_training_data
    model = RandomForestModel()
    model.fit(X_train, y_train)

    monkeypatch.setattr(model._model, "classes_", np.array([1, -1, 0]))
    fake_raw_proba = np.array([[0.2, 0.5, 0.3]])  # BUY=0.2, SELL=0.5, HOLD=0.3
    monkeypatch.setattr(model._model, "predict_proba", lambda X: fake_raw_proba)

    result = model.predict_proba(X_train.iloc[[0]])

    assert list(result.columns) == CLASS_NAMES
    assert result.iloc[0]["SELL"] == pytest.approx(0.5)
    assert result.iloc[0]["HOLD"] == pytest.approx(0.3)
    assert result.iloc[0]["BUY"] == pytest.approx(0.2)


def test_get_feature_importances_returns_all_features_summing_to_one(
    small_training_data,
) -> None:
    X_train, y_train = small_training_data
    model = RandomForestModel()
    model.fit(X_train, y_train)

    importances = model.get_feature_importances()

    assert set(importances.keys()) == set(FEATURE_COLUMNS)
    assert sum(importances.values()) == pytest.approx(1.0)