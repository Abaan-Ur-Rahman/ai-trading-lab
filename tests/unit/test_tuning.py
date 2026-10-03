"""Unit tests for tune_hyperparameters / TuningResult."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import tuning
from ml.dataset import FEATURE_COLUMNS
from ml.models.logistic_regression import LogisticRegressionModel
from ml.models.random_forest import RandomForestModel
from ml.tuning import tune_hyperparameters


@pytest.fixture
def synthetic_train_partition() -> pd.DataFrame:
    """A dataset shaped like a TrainingResult.train partition.

    300 rows is enough for TimeSeriesSplit(n_splits=3, gap=horizon) to
    carve out non-trivial folds while keeping the test fast.
    """
    rng = np.random.default_rng(0)
    n = 300

    features = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    labels = np.tile([-1, 0, 1], n // 3 + 1)[:n]

    return pd.DataFrame(
        {
            **features,
            "label": labels,
            "close": 100.0 + rng.normal(size=n).cumsum(),
        }
    )


def test_tune_hyperparameters_selects_best_and_records_all_trials(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """Every grid combination should be recorded, and the winner actually returned."""
    result = tune_hyperparameters(
        model_factory=LogisticRegressionModel,
        param_grid={"C": [0.1, 1.0, 10.0]},
        fixed_params={"class_weight": "balanced", "random_state": 42},
        train=synthetic_train_partition,
        horizon=2,
        n_splits=3,
    )

    assert len(result.trials) == 3
    assert {trial.hyperparameters["C"] for trial in result.trials} == {0.1, 1.0, 10.0}
    assert result.best_hyperparameters in [trial.hyperparameters for trial in result.trials]
    assert result.best_mean_macro_f1 == max(trial.mean_macro_f1 for trial in result.trials)

    predictions = result.model.predict(synthetic_train_partition[FEATURE_COLUMNS])
    assert set(np.unique(predictions)).issubset({-1, 0, 1})


def test_tune_hyperparameters_refits_winner_on_full_train(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """The returned scaler must be fit on the full train partition, not a CV fold."""
    result = tune_hyperparameters(
        model_factory=LogisticRegressionModel,
        param_grid={"C": [0.1, 1.0]},
        fixed_params={"random_state": 42},
        train=synthetic_train_partition,
        horizon=2,
        n_splits=3,
    )

    expected_mean = synthetic_train_partition[FEATURE_COLUMNS].mean()
    pd.testing.assert_series_equal(result.scaler.mean, expected_mean, check_names=False)


def test_tune_hyperparameters_fits_scaler_separately_per_fold(
    synthetic_train_partition: pd.DataFrame, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scaler must be fit fresh inside each CV fold, not once on the whole
    train partition before cross-validation -- fitting once upfront would
    leak each fold's held-out rows' statistics into every other fold's
    scaling.
    """
    fit_calls: list[int] = []
    original_fit_scaler = tuning.fit_scaler

    def recording_fit_scaler(training_features: pd.DataFrame):
        fit_calls.append(len(training_features))
        return original_fit_scaler(training_features)

    monkeypatch.setattr(tuning, "fit_scaler", recording_fit_scaler)

    tune_hyperparameters(
        model_factory=LogisticRegressionModel,
        param_grid={"C": [0.1, 1.0]},
        fixed_params={"random_state": 42},
        train=synthetic_train_partition,
        horizon=2,
        n_splits=3,
    )

    # 2 combinations x 3 folds = 6 per-fold fits, plus 1 final full-train refit.
    assert len(fit_calls) == 2 * 3 + 1
    *fold_calls, final_call = fit_calls
    assert all(n < len(synthetic_train_partition) for n in fold_calls)
    assert final_call == len(synthetic_train_partition)


def test_tune_hyperparameters_works_with_random_forest(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """The function must be model-agnostic, not LogisticRegression-specific."""
    result = tune_hyperparameters(
        model_factory=RandomForestModel,
        param_grid={"n_estimators": [50, 100], "max_depth": [5, 10]},
        fixed_params={"min_samples_leaf": 20, "class_weight": "balanced", "random_state": 42},
        train=synthetic_train_partition,
        horizon=2,
        n_splits=3,
    )

    assert len(result.trials) == 4  # 2 n_estimators x 2 max_depth
    assert result.best_hyperparameters["n_estimators"] in [50, 100]
    assert result.best_hyperparameters["max_depth"] in [5, 10]


def test_tune_hyperparameters_does_not_mutate_input(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """tune_hyperparameters must not modify the train partition passed in."""
    original = synthetic_train_partition.copy(deep=True)

    tune_hyperparameters(
        model_factory=LogisticRegressionModel,
        param_grid={"C": [0.1, 1.0]},
        fixed_params={"random_state": 42},
        train=synthetic_train_partition,
        horizon=2,
        n_splits=3,
    )

    pd.testing.assert_frame_equal(synthetic_train_partition, original)


def test_tune_hyperparameters_rejects_too_few_splits(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """n_splits must be at least 2."""
    with pytest.raises(ValueError):
        tune_hyperparameters(
            model_factory=LogisticRegressionModel,
            param_grid={"C": [1.0]},
            fixed_params={"random_state": 42},
            train=synthetic_train_partition,
            horizon=2,
            n_splits=1,
        )


def test_tune_hyperparameters_rejects_non_positive_horizon(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """horizon must be greater than 0."""
    with pytest.raises(ValueError):
        tune_hyperparameters(
            model_factory=LogisticRegressionModel,
            param_grid={"C": [1.0]},
            fixed_params={"random_state": 42},
            train=synthetic_train_partition,
            horizon=0,
            n_splits=3,
        )


def test_tune_hyperparameters_rejects_empty_param_grid(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """An empty param_grid defeats the point of tuning and should raise."""
    with pytest.raises(ValueError):
        tune_hyperparameters(
            model_factory=LogisticRegressionModel,
            param_grid={},
            fixed_params={"random_state": 42},
            train=synthetic_train_partition,
            horizon=2,
            n_splits=3,
        )


def test_tune_hyperparameters_rejects_empty_value_list(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """A param_grid entry with no values to try should raise."""
    with pytest.raises(ValueError):
        tune_hyperparameters(
            model_factory=LogisticRegressionModel,
            param_grid={"C": []},
            fixed_params={"random_state": 42},
            train=synthetic_train_partition,
            horizon=2,
            n_splits=3,
        )


def test_tune_hyperparameters_rejects_overlapping_keys(
    synthetic_train_partition: pd.DataFrame,
) -> None:
    """param_grid and fixed_params must not both specify the same hyperparameter."""
    with pytest.raises(ValueError):
        tune_hyperparameters(
            model_factory=LogisticRegressionModel,
            param_grid={"C": [1.0]},
            fixed_params={"C": 0.5, "random_state": 42},
            train=synthetic_train_partition,
            horizon=2,
            n_splits=3,
        )