"""Unit tests for train_model / TrainingResult."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.dataset import FEATURE_COLUMNS
from ml.models.logistic_regression import LogisticRegressionModel
from ml.training import train_model


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    """A dataset shaped like build_feature_dataset's output.

    Includes FEATURE_COLUMNS, a 'label' column, and an extra 'close'
    column that is NOT a feature -- standing in for the OHLCV columns
    that survive the feature pipeline for later trading evaluation.
    """
    rng = np.random.default_rng(0)
    n = 300

    features = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    # Repeating block of all three classes so every chronological slice
    # (train/val/test) contains all three labels.
    labels = np.tile([-1, 0, 1], n // 3 + 1)[:n]

    data = pd.DataFrame(
        {
            **features,
            "label": labels,
            "close": 100.0 + rng.normal(size=n).cumsum(),
        }
    )
    return data


def test_train_model_returns_fitted_model(synthetic_dataset: pd.DataFrame) -> None:
    """The returned model should already be fit (predict works, no raise)."""
    model = LogisticRegressionModel()

    result = train_model(model, synthetic_dataset, horizon=2)

    predictions = result.model.predict(result.train[FEATURE_COLUMNS])
    assert set(np.unique(predictions)).issubset({-1, 0, 1})


def test_train_model_scaler_fit_only_on_train(synthetic_dataset: pd.DataFrame) -> None:
    """The scaler's mean/std must match the train partition's features, not val/test."""
    model = LogisticRegressionModel()

    result = train_model(model, synthetic_dataset, horizon=2)

    expected_mean = result.train[FEATURE_COLUMNS].mean()
    pd.testing.assert_series_equal(result.scaler.mean, expected_mean, check_names=False)


def test_train_model_partitions_are_unscaled_and_keep_all_columns(
    synthetic_dataset: pd.DataFrame,
) -> None:
    """train/val/test must retain non-feature columns (e.g. close) and raw values."""
    model = LogisticRegressionModel()

    result = train_model(model, synthetic_dataset, horizon=2)

    assert "close" in result.train.columns
    assert "close" in result.val.columns
    assert "close" in result.test.columns

    # Raw (unscaled) values: train partition's features should match the
    # original dataset's values at the same rows, not scaler-transformed values.
    original_slice = synthetic_dataset.loc[result.train.index, FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(result.train[FEATURE_COLUMNS], original_slice)


def test_train_model_does_not_mutate_input_dataset(synthetic_dataset: pd.DataFrame) -> None:
    """train_model must not modify the dataset passed in."""
    original = synthetic_dataset.copy()
    model = LogisticRegressionModel()

    train_model(model, synthetic_dataset, horizon=2)

    pd.testing.assert_frame_equal(synthetic_dataset, original)


def test_train_model_raises_when_training_partition_missing_a_class(
    synthetic_dataset: pd.DataFrame,
) -> None:
    """A training partition missing a required class should raise ValueError."""
    only_hold = synthetic_dataset.copy()
    only_hold["label"] = 0  # every row HOLD -> train partition missing SELL/BUY
    model = LogisticRegressionModel()

    with pytest.raises(ValueError):
        train_model(model, only_hold, horizon=2)


def test_train_model_propagates_invalid_split_config(synthetic_dataset: pd.DataFrame) -> None:
    """Invalid train_pct/val_pct should propagate from chronological_split."""
    model = LogisticRegressionModel()

    with pytest.raises(ValueError):
        train_model(model, synthetic_dataset, horizon=2, train_pct=0.9, val_pct=0.3)