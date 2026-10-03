"""Hyperparameter tuning via chronological cross-validation on training data only.

Tuning never touches validation or test. Each hyperparameter combination
is scored by averaging macro-F1 across TimeSeriesSplit folds carved out of
the `train` partition alone, with a purge gap (sklearn's `gap` parameter)
between each fold's training and held-out rows -- the same leakage concern
chronological_split's purge gap addresses, reused here via TimeSeriesSplit
rather than reimplemented. Once the best combination is selected, a fresh
model is retrained on the *entire* train partition (still never touching
val/test) using those hyperparameters -- that retrained model and its
scaler are what the rest of the pipeline consumes.

The scaler is fit separately inside every fold using only that fold's
training rows, never on the whole train partition before cross-validation
-- fitting once upfront would leak each fold's held-out statistics into
every other fold's scaling parameters.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from features.scaling import FeatureScaler, apply_scaler, fit_scaler
from ml.dataset import CLASS_LABELS, separate_features_and_target, validate_all_classes_present
from ml.models.base import ModelWrapper


@dataclass
class TuningTrial:
    """One hyperparameter combination's cross-validated score.

    `hyperparameters` holds only the tuned subset (the keys from
    param_grid) -- fixed_params are constant across every trial, so
    repeating them in each one would just be noise when inspecting results.
    """

    hyperparameters: dict
    fold_scores: list[float]
    mean_macro_f1: float


@dataclass
class TuningResult:
    """Full record of a tuning run: every trial tried, plus the retrained winner.

    `model` and `scaler` are fit on the ENTIRE train partition using
    `best_hyperparameters` -- never on a CV fold subset, and never on
    val/test. Call `model.get_hyperparameters()` for the complete
    hyperparameter set including fixed_params; `best_hyperparameters`
    here is only the tuned subset, matching `trials` entries.
    """

    trials: list[TuningTrial]
    best_hyperparameters: dict
    best_mean_macro_f1: float
    model: ModelWrapper
    scaler: FeatureScaler


def _expand_param_grid(param_grid: dict[str, list]) -> list[dict]:
    """Expand a {param_name: [values, ...]} grid into individual combinations."""
    names = list(param_grid.keys())
    value_lists = [param_grid[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def tune_hyperparameters(
    model_factory: Callable[..., ModelWrapper],
    param_grid: dict[str, list],
    fixed_params: dict,
    train: pd.DataFrame,
    horizon: int,
    feature_columns: list[str] | None = None,
    n_splits: int = 3,
) -> TuningResult:
    """Select hyperparameters via chronological CV on `train`, then refit on all of it.

    Does not mutate `train`. Never reads val or test -- this function only
    ever receives the train partition in the first place.

    Args:
        model_factory: A ModelWrapper subclass (e.g. LogisticRegressionModel,
            RandomForestModel) called as model_factory(**combo, **fixed_params)
            to construct a fresh, unfit model for each hyperparameter
            combination and for the final refit.
        param_grid: {hyperparameter_name: [values to try]}. Every
            combination (Cartesian product across all keys) is tried.
        fixed_params: Hyperparameters held constant across every
            combination (e.g. {"class_weight": "balanced", "random_state": 42}).
            Must not share any keys with param_grid.
        train: The training partition only (e.g. TrainingResult.train),
            still carrying every original column -- feature selection
            happens inside this function via feature_columns.
        horizon: The label horizon the dataset was built with. Used as
            TimeSeriesSplit's `gap`, so no CV fold's training rows have a
            label window reaching into that fold's held-out rows.
        feature_columns: Forwarded to separate_features_and_target for
            every fold and the final refit. Defaults to FEATURE_COLUMNS
            when not provided.
        n_splits: Number of chronological CV folds carved out of `train`.
            Kept small deliberately -- this is model/hyperparameter
            selection, not an aggressive search.

    Returns:
        A TuningResult with every trial's scores, the winning
        hyperparameters, and a model+scaler retrained on the full train
        partition.

    Raises:
        ValueError: If n_splits < 2, horizon <= 0, param_grid is empty or
            contains an empty value list, or fixed_params and param_grid
            share a key. Propagated from sklearn's TimeSeriesSplit if
            n_splits/horizon are incompatible with len(train). Propagated
            from validate_all_classes_present if a fold's (or the full
            train partition's) training rows are missing a required class.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if horizon <= 0:
        raise ValueError("horizon must be greater than 0")

    if not param_grid:
        raise ValueError("param_grid must not be empty")

    for name, values in param_grid.items():
        if not values:
            raise ValueError(f"param_grid['{name}'] must not be empty")

    overlapping_keys = set(param_grid).intersection(fixed_params)
    if overlapping_keys:
        raise ValueError(
            f"param_grid and fixed_params must not share keys: {sorted(overlapping_keys)}",
        )

    combinations = _expand_param_grid(param_grid)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)

    trials: list[TuningTrial] = []

    for combo in combinations:
        fold_scores: list[float] = []

        for fold_train_idx, fold_val_idx in splitter.split(train):
            fold_train = train.iloc[fold_train_idx]
            fold_val = train.iloc[fold_val_idx]

            X_fold_train, y_fold_train = separate_features_and_target(
                fold_train, feature_columns=feature_columns,
            )
            X_fold_val, y_fold_val = separate_features_and_target(
                fold_val, feature_columns=feature_columns,
            )

            validate_all_classes_present(y_fold_train)

            # Fit fresh on THIS fold's training rows only -- see module
            # docstring for why fitting once on the whole train partition
            # upfront would leak across folds.
            fold_scaler = fit_scaler(X_fold_train)
            X_fold_train_scaled = apply_scaler(fold_scaler, X_fold_train)
            X_fold_val_scaled = apply_scaler(fold_scaler, X_fold_val)

            model = model_factory(**combo, **fixed_params)
            model.fit(X_fold_train_scaled, y_fold_train)

            y_fold_pred = model.predict(X_fold_val_scaled)
            fold_macro_f1 = f1_score(
                y_fold_val, y_fold_pred, labels=CLASS_LABELS, average="macro", zero_division=0,
            )
            fold_scores.append(float(fold_macro_f1))

        trials.append(
            TuningTrial(
                hyperparameters=combo,
                fold_scores=fold_scores,
                mean_macro_f1=sum(fold_scores) / len(fold_scores),
            ),
        )

    best_trial = max(trials, key=lambda trial: trial.mean_macro_f1)

    # Retrain on the FULL train partition with the winning hyperparameters.
    # The CV folds above only ever see subsets of train -- this final fit
    # still never touches val or test.
    X_train, y_train = separate_features_and_target(train, feature_columns=feature_columns)
    validate_all_classes_present(y_train)

    final_scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(final_scaler, X_train)

    final_model = model_factory(**best_trial.hyperparameters, **fixed_params)
    final_model.fit(X_train_scaled, y_train)

    return TuningResult(
        trials=trials,
        best_hyperparameters=best_trial.hyperparameters,
        best_mean_macro_f1=best_trial.mean_macro_f1,
        model=final_model,
        scaler=final_scaler,
    )