"""Unit tests for chronological dataset splitting and feature/target separation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.dataset import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    chronological_split,
    separate_features_and_target,
    validate_all_classes_present,
)


def _make_feature_dataset(rows: int = 50, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic feature+label dataset shaped like build_feature_dataset's output."""
    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "log_return": rng.standard_normal(rows),
            "ema_gap_pct": rng.standard_normal(rows),
            "rsi": rng.uniform(0, 100, rows),
            "atr_pct": rng.uniform(0, 0.05, rows),
            "macd_hist_pct": rng.standard_normal(rows),
            "label": rng.choice([-1, 0, 1], size=rows),
        }
    )


def test_chronological_split_produces_expected_row_counts() -> None:
    """Partition sizes should follow train_pct/val_pct before purging, minus the purge gap."""
    dataset = _make_feature_dataset(rows=50)

    train, val, test = chronological_split(dataset, horizon=3, train_pct=0.6, val_pct=0.2)

    assert len(train) == 30
    assert len(val) == 7
    assert len(test) == 7


def test_chronological_split_purges_exactly_horizon_rows_at_each_boundary() -> None:
    """No training row's label window should reach into validation, and likewise for val/test."""
    dataset = _make_feature_dataset(rows=50)
    horizon = 3

    train, val, test = chronological_split(dataset, horizon=horizon, train_pct=0.6, val_pct=0.2)

    # The purge gap means val must start strictly after train_end + horizon.
    assert val.index.min() > train.index.max() + horizon - 1
    assert val.index.min() - train.index.max() == horizon + 1

    assert test.index.min() > val.index.max() + horizon - 1
    assert test.index.min() - val.index.max() == horizon + 1


def test_chronological_split_partitions_are_non_overlapping_and_ordered() -> None:
    """Train, val, and test indices must be strictly increasing and never overlap."""
    dataset = _make_feature_dataset(rows=50)

    train, val, test = chronological_split(dataset, horizon=3, train_pct=0.6, val_pct=0.2)

    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_chronological_split_rejects_non_positive_horizon() -> None:
    """horizon must be greater than 0."""
    dataset = _make_feature_dataset(rows=50)

    with pytest.raises(ValueError):
        chronological_split(dataset, horizon=0)


def test_chronological_split_rejects_invalid_train_pct() -> None:
    """train_pct must be strictly between 0 and 1."""
    dataset = _make_feature_dataset(rows=50)

    with pytest.raises(ValueError):
        chronological_split(dataset, horizon=3, train_pct=1.0)


def test_chronological_split_rejects_invalid_val_pct() -> None:
    """val_pct must be strictly between 0 and 1."""
    dataset = _make_feature_dataset(rows=50)

    with pytest.raises(ValueError):
        chronological_split(dataset, horizon=3, val_pct=0.0)


def test_chronological_split_rejects_percentages_leaving_no_test_set() -> None:
    """train_pct + val_pct must leave room for a non-empty test set."""
    dataset = _make_feature_dataset(rows=50)

    with pytest.raises(ValueError):
        chronological_split(dataset, horizon=3, train_pct=0.6, val_pct=0.4)


def test_chronological_split_rejects_empty_partition_from_excessive_purging() -> None:
    """A horizon large enough to purge an entire partition should raise, not return empty."""
    dataset = _make_feature_dataset(rows=20)

    with pytest.raises(ValueError):
        chronological_split(dataset, horizon=15, train_pct=0.5, val_pct=0.25)


def test_chronological_split_does_not_mutate_input() -> None:
    """chronological_split must not modify the caller's original dataframe."""
    dataset = _make_feature_dataset(rows=50)
    original = dataset.copy(deep=True)

    chronological_split(dataset, horizon=3, train_pct=0.6, val_pct=0.2)

    pd.testing.assert_frame_equal(dataset, original)


def test_separate_features_and_target_returns_expected_columns() -> None:
    """X should contain exactly FEATURE_COLUMNS; y should be the int-cast label."""
    dataset = _make_feature_dataset(rows=10)

    X, y = separate_features_and_target(dataset)

    assert list(X.columns) == FEATURE_COLUMNS
    assert "label" not in X.columns
    assert y.name == "label"
    assert y.dtype == np.int64 or y.dtype == int


def test_separate_features_and_target_requires_all_feature_columns() -> None:
    """Missing a required feature column should raise ValueError."""
    dataset = pd.DataFrame({"log_return": [0.1, 0.2], "label": [0, 1]})

    with pytest.raises(ValueError):
        separate_features_and_target(dataset)


def test_separate_features_and_target_requires_label_column() -> None:
    """Missing the label column should raise ValueError."""
    dataset = pd.DataFrame({col: [0.0, 0.1] for col in FEATURE_COLUMNS})

    with pytest.raises(ValueError):
        separate_features_and_target(dataset)


def test_separate_features_and_target_does_not_mutate_input() -> None:
    """separate_features_and_target must not modify the caller's original dataframe."""
    dataset = _make_feature_dataset(rows=10)
    original = dataset.copy(deep=True)

    separate_features_and_target(dataset)

    pd.testing.assert_frame_equal(dataset, original)


def test_validate_all_classes_present_passes_when_all_classes_exist() -> None:
    """No exception should be raised when y_train contains every class in CLASS_LABELS."""
    y_train = pd.Series(CLASS_LABELS * 5)

    validate_all_classes_present(y_train)  # should not raise


def test_validate_all_classes_present_raises_when_a_class_is_missing() -> None:
    """A y_train missing a class (e.g. all-HOLD) should raise a clear ValueError."""
    y_train = pd.Series([0, 0, 0, 0])

    with pytest.raises(ValueError):
        validate_all_classes_present(y_train)