"""Training orchestration for the ML layer.

Wires together dataset splitting, scaling, and model fitting into a single
entry point. Downstream evaluation and persistence consume this function's
TrainingResult rather than re-deriving splits or scalers themselves, so
there is exactly one place where "how was this model trained" is decided.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.scaling import FeatureScaler, apply_scaler, fit_scaler
from ml.dataset import (
    chronological_split,
    separate_features_and_target,
    validate_all_classes_present,
)
from ml.models.base import ModelWrapper


@dataclass
class TrainingResult:
    """Everything produced by training a model, needed for evaluation/persistence.

    `train`, `val`, and `test` are the UNSCALED chronological-split
    partitions, with every original dataset column intact (not just
    FEATURE_COLUMNS) -- e.g. they still carry `close`, which trading
    evaluation needs later but the model itself never sees. Scaling is
    applied only to training features, inside this module; callers
    (evaluation, inference) apply `scaler` to val/test themselves at the
    point they actually need scaled features, the same way a real
    inference call would.
    """

    model: ModelWrapper
    scaler: FeatureScaler
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def train_model(
    model: ModelWrapper,
    dataset: pd.DataFrame,
    horizon: int,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> TrainingResult:
    """Split, scale, and fit `model` on `dataset`.

    `model` is mutated in place (fit() is called on it) and is also
    returned inside TrainingResult for clarity at the call site.

    The scaler is fit ONLY on the training partition's features, never on
    val or test, to avoid leaking their distribution into training.

    Does not mutate `dataset`.

    Args:
        model: An unfit ModelWrapper to train (e.g. a freshly constructed
            LogisticRegressionModel with whatever class_weight you want).
        dataset: Chronologically-ordered feature+label data, e.g. the
            output of build_feature_dataset.
        horizon: The label horizon the dataset was built with (passed
            through to chronological_split for purging).
        train_pct: Fraction of rows allocated to training.
        val_pct: Fraction of rows allocated to validation.

    Returns:
        A TrainingResult with the fitted model, fitted scaler, and the
        three unscaled split partitions.

    Raises:
        ValueError: Propagated from chronological_split (invalid split
            config or an empty partition after purging) or from
            validate_all_classes_present (training partition missing a
            required class).
    """
    train, val, test = chronological_split(
        dataset,
        horizon=horizon,
        train_pct=train_pct,
        val_pct=val_pct,
    )

    X_train, y_train = separate_features_and_target(train)
    validate_all_classes_present(y_train)

    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(scaler, X_train)

    model.fit(X_train_scaled, y_train)

    return TrainingResult(model=model, scaler=scaler, train=train, val=val, test=test)